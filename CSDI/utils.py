import math
import torch
from torch.optim import Adam
from tqdm import tqdm


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    resume_state=None,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    max_train_steps = config.get("max_train_steps", None)
    start_global_step = 0
    start_epoch = 0
    if resume_state is not None:
        start_global_step = int(resume_state.get("global_step", 0) or 0)
        start_epoch = int(resume_state.get("epoch_no", 0) or 0)
        optimizer_state = resume_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            resume_state["optimizer_state_dict"] = None

    total_epochs = config["epochs"]
    if max_train_steps is not None:
        steps_per_epoch = max(1, len(train_loader))
        remaining_steps = max(0, max_train_steps - start_global_step)
        total_epochs = max(1, math.ceil(remaining_steps / steps_per_epoch))

    p1 = int(0.75 * total_epochs)
    p2 = int(0.9 * total_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    if resume_state is not None:
        scheduler_state = resume_state.get("scheduler_state_dict")
        if scheduler_state is not None:
            lr_scheduler.load_state_dict(scheduler_state)
            resume_state["scheduler_state_dict"] = None

    best_valid_loss = 1e10
    global_step = start_global_step
    save_interval_steps = config.get("save_interval_steps", None)
    epoch_no = start_epoch
    while True:
        if max_train_steps is not None and global_step >= max_train_steps:
            print("\n max_train_steps reached:", global_step)
            break
        if max_train_steps is None and epoch_no >= config["epochs"]:
            break
        avg_loss = 0
        model.train()
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_no)
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                if loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                global_step += 1
                if (
                    foldername != ""
                    and save_interval_steps is not None
                    and global_step % save_interval_steps == 0
                ):
                    checkpoint_state = {
                        "format_version": 1,
                        "model_state_dict": _unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": lr_scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch_no": epoch_no,
                    }
                    torch.save(
                        checkpoint_state,
                        foldername + "/checkpoint_step_" + str(global_step) + ".pth",
                    )
                    torch.save(checkpoint_state, foldername + "/training_state.pth")
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                        "global_step": global_step,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break
                if max_train_steps is not None and global_step >= max_train_steps:
                    break

            lr_scheduler.step()
        if max_train_steps is not None and global_step >= max_train_steps:
            print("\n max_train_steps reached:", global_step)
            break
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        if loss.dim() > 0:
                            loss = loss.mean()
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
        epoch_no += 1

    if foldername != "":
        checkpoint_state = {
            "format_version": 1,
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "global_step": global_step,
            "epoch_no": epoch_no,
        }
        torch.save(checkpoint_state, foldername + "/training_state.pth")
        torch.save(_unwrap_model(model).state_dict(), output_path)
