import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
CSDI_DIR = REPO_ROOT / "CSDI"
sys.path.insert(0, str(CSDI_DIR))


ARKIT_52_NAMES = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]


IMAGE_PATHS = [
    "outputs/states/000.png",
    "outputs/states/001.png",
    "outputs/states/002.png",
]


BLENDSHAPE_KEYFRAMES = np.array(
    [
        [
            0.6389522552490234,
            0.6411657929420471,
            0.00029021064983680844,
            0.0006118063465692103,
            0.0008069037576206028,
            2.0384382878546603e-05,
            1.49640953850394e-07,
            2.470166009516106e-07,
            0.13096295297145844,
            0.1261022537946701,
            0.08604716509580612,
            0.09868430346250534,
            0.024840811267495155,
            0.0978197231888771,
            0.14576587080955505,
            0.05172456428408623,
            0.10695620626211166,
            0.09258322417736053,
            0.5946202278137207,
            0.4704403281211853,
            0.003661601571366191,
            0.0022283028811216354,
            5.922576383454725e-05,
            0.0006133257993496954,
            0.007505400106310844,
            3.1127849069889635e-05,
            0.004771846812218428,
            0.015069358050823212,
            0.006752856075763702,
            0.0011188649805262685,
            0.001323020551353693,
            0.00022146262926980853,
            0.0013231548946350813,
            0.00010610248136799783,
            0.000101469959190581,
            0.20596736669540405,
            0.12466926872730255,
            0.001302312477491796,
            0.000661318888887763,
            0.02869098260998726,
            0.009032546542584896,
            0.06982415169477463,
            0.010583004914224148,
            0.009524846449494362,
            0.008955707773566246,
            0.004972450435161591,
            0.006722633261233568,
            0.00011695056309690699,
            0.00019117703777737916,
            2.0970966829736426e-07,
            7.712154683758854e-07,
            0.0,
        ],
        [
            0.6595447659492493,
            0.6647862195968628,
            0.0005861804238520563,
            0.0005747302784584463,
            0.0007497536134906113,
            3.77539181499742e-05,
            1.9932393513499846e-07,
            1.5937769148877123e-07,
            0.14731480181217194,
            0.13492949306964874,
            0.08664200454950333,
            0.09982400387525558,
            0.023395905271172523,
            0.11813386529684067,
            0.16271501779556274,
            0.05296614021062851,
            0.10493993759155273,
            0.10756310820579529,
            0.6009368896484375,
            0.5293713808059692,
            0.0037228562869131565,
            0.0030256262980401516,
            8.90860246727243e-05,
            0.0008055146900005639,
            0.004537018947303295,
            1.694069396762643e-05,
            0.0018962068716064095,
            0.009166779927909374,
            0.005477591417729855,
            0.00014698792074341327,
            0.00014477840159088373,
            0.0007810455863364041,
            0.00578073738142848,
            0.0001094877952709794,
            0.00013804079208057374,
            0.0480303019285202,
            0.03781750053167343,
            0.01115835178643465,
            0.00023526052245870233,
            0.008853207342326641,
            0.012635372579097748,
            0.019623888656497,
            0.006022143643349409,
            0.7178910970687866,
            0.661141037940979,
            0.005138728767633438,
            0.007094150874763727,
            0.0004003327339887619,
            0.0006197435432113707,
            1.7127321427778952e-07,
            9.779087122296914e-07,
            0.0,
        ],
        [
            0.7121870517730713,
            0.6806215643882751,
            0.0007876614108681679,
            0.00048239610623568296,
            0.001248133834451437,
            3.893935354426503e-05,
            5.961362035122875e-07,
            4.491161007535993e-07,
            0.0983935296535492,
            0.07415531575679779,
            0.15983998775482178,
            0.1758294254541397,
            0.8303061127662659,
            0.0006389931077137589,
            0.0015348391607403755,
            0.8272788524627686,
            0.05019678920507431,
            0.049552131444215775,
            0.6702951192855835,
            0.49618831276893616,
            0.005667698569595814,
            0.008278865367174149,
            0.00011445347627159208,
            0.0005640859017148614,
            0.003799069207161665,
            3.829749039141461e-05,
            0.0011587085900828242,
            0.022218724712729454,
            0.0034479694440960884,
            0.0003154874430038035,
            0.0003551999689079821,
            0.0009090170497074723,
            0.008398128673434258,
            0.00015665461251046509,
            0.0002388817520113662,
            0.061241768300533295,
            0.03744148090481758,
            0.010285690426826477,
            0.00011479187378427014,
            0.011399514973163605,
            0.01301783137023449,
            0.0195947103202343,
            0.005631303880363703,
            0.6961560249328613,
            0.621907651424408,
            0.0053013949654996395,
            0.013570432551205158,
            0.0005693918792530894,
            0.0005431175814010203,
            1.6632223776014143e-07,
            2.5115220978477737e-06,
            0.0,
        ],
    ],
    dtype=np.float32,
)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path):
    import yaml

    config_path = resolve_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(config, checkpoint_path, device):
    import torch
    from main_model import CSDI_Express4D

    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = CSDI_Express4D(
        config,
        device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if all(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def infer_keyframe_sequence(model, keyframes, duration, num_samples):
    import torch

    if keyframes.ndim != 2 or keyframes.shape[1] != 52:
        raise ValueError(f"keyframes must have shape [N,52], got {keyframes.shape}")
    if len(keyframes) < 2:
        raise ValueError("At least two keyframes are required")

    keyframes = np.nan_to_num(keyframes, nan=0.0, posinf=0.0, neginf=0.0)
    keyframes = np.clip(keyframes, 0.0, 1.0).astype(np.float32)

    full_sequence = []
    frame_meta = []
    segment_middles = []

    for segment_index in range(len(keyframes) - 1):
        start_np = keyframes[segment_index]
        end_np = keyframes[segment_index + 1]

        start = torch.from_numpy(start_np)
        end = torch.from_numpy(end_np)
        with torch.no_grad():
            middle = model.generate_middle(start, end, duration, num_samples=num_samples)
        middle_np = middle.detach().cpu().numpy().astype(np.float32)

        if num_samples != 1:
            raise ValueError("This script writes one final sequence; keep --num_samples 1")

        middle_np = middle_np[0]
        segment_middles.append(middle_np)
        segment_full = np.concatenate([start_np[None], middle_np, end_np[None]], axis=0)

        first_frame = 0 if segment_index == 0 else 1
        for local_index, values in enumerate(segment_full[first_frame:], start=first_frame):
            full_sequence.append(values)
            if local_index == 0:
                kind = "keyframe_start"
                keyframe_index = segment_index
            elif local_index == len(segment_full) - 1:
                kind = "keyframe_end"
                keyframe_index = segment_index + 1
            else:
                kind = "generated_middle"
                keyframe_index = None
            frame_meta.append(
                {
                    "frame_index": len(frame_meta),
                    "segment_index": segment_index,
                    "segment_local_index": local_index,
                    "kind": kind,
                    "keyframe_index": keyframe_index,
                }
            )

    return np.stack(full_sequence, axis=0).astype(np.float32), segment_middles, frame_meta


def save_csv(path, sequence):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + ARKIT_52_NAMES)
        for index, row in enumerate(sequence):
            writer.writerow([index] + [float(value) for value in row])


def main():
    parser = argparse.ArgumentParser(description="Infer Express4D sequence from embedded keyframes.")
    parser.add_argument("--config", default="CSDI/config/express4d.yaml")
    parser.add_argument(
        "--checkpoint",
        default="save/express4d_20260528_023203/checkpoint_step_10000.pth",
    )
    parser.add_argument("--duration", "--duraction", dest="duration", type=float, default=3.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs/keyframe_infer")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = load_config(args.config)
    model = load_model(config, args.checkpoint, device)

    sequence, segment_middles, frame_meta = infer_keyframe_sequence(
        model,
        BLENDSHAPE_KEYFRAMES,
        duration=args.duration,
        num_samples=args.num_samples,
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "keyframes.npy", BLENDSHAPE_KEYFRAMES)
    np.save(output_dir / "generated_sequence.npy", sequence)
    save_csv(output_dir / "generated_sequence.csv", sequence)

    for index, middle in enumerate(segment_middles):
        np.save(output_dir / f"segment_{index:03d}_{index + 1:03d}_middle.npy", middle)

    metadata = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "config": str(resolve_path(args.config)),
        "duration": args.duration,
        "duration_unit": "seconds",
        "duration_usage": "passed to each adjacent keyframe segment",
        "image_paths": IMAGE_PATHS,
        "arkit_52_names": ARKIT_52_NAMES,
        "input_keyframes_shape": list(BLENDSHAPE_KEYFRAMES.shape),
        "generated_sequence_shape": list(sequence.shape),
        "frame_meta": frame_meta,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved generated sequence: {output_dir / 'generated_sequence.npy'} {sequence.shape}")
    print(f"Saved generated CSV:      {output_dir / 'generated_sequence.csv'}")
    print(f"Saved metadata:           {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
