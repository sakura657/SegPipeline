# config_prompt.py

# —— Task Description —— 
task_description = """In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X‑ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR‑Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day. In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x‑ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time‑consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment.

The UW‑Madison Carbone Cancer Center is a pioneer in MR‑Linac based radiotherapy, and has treated patients with MRI guided radiotherapy based on their daily anatomy since 2015. UW‑Madison has generously agreed to support this project which provides anonymized MRIs of patients treated at the UW‑Madison Carbone Cancer Center. The University of Wisconsin‑Madison is a public land‑grant research university in Madison, Wisconsin. The Wisconsin Idea is the university's pledge to the state, the nation, and the world that their endeavors will benefit all citizens.

In this competition, you’ll create a model to automatically segment the stomach and intestines on MRI scans. The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment. You'll base your algorithm on a dataset of these scans to come up with creative deep learning solutions that will help cancer patients get better care.

Cancer takes enough of a toll. If successful, you'll enable radiation oncologists to safely deliver higher doses of radiation to tumors while avoiding the stomach and intestines. This will make cancer patients' daily treatments faster and allow them to get more effective treatment with less side effects and better long-term cancer control.
"""

# —— Evaluation Metrics —— 
evaluation = """This competition is evaluated on the mean Dice coefficient and 3D Hausdorff distance.

- Dice coefficient: Measures pixel-wise agreement between a predicted segmentation and its ground truth, defined as:
  2 * |X ∩ Y| / (|X| + |Y|),
  where X is the predicted set and Y is the ground truth. It is 0 when both X and Y are empty. The leaderboard score is the mean Dice across all test images.

- 3D Hausdorff distance: Computes the greatest distance from a point in one segmentation to the closest point in the other. For 3D volumes, slices are stacked with a depth of 1mm. The expected/predicted locations are normalized by image size to yield a bounded 0–1 score.

The final leaderboard score combines the two metrics with weights: 0.4 for Dice and 0.6 for Hausdorff distance.
"""

# —— Dataset Description —— 
dataset_description = """In this competition we are segmenting organs in medical images. The training annotations are provided as RLE-encoded masks, and the images are 16-bit grayscale PNGs.

Each case consists of multiple scan slice sets (one per day). Some cases split early days into train and later days into test; others keep entire cases in train or test. The hidden test set contains ~50 cases with varying days and slices, accessible only upon submission. A sample_submission.csv placeholder shows required format; your code must generate the actual submission CSV against the hidden test set.

Files:
- train.csv: IDs and RLE-encoded masks for all training objects.
- sample_submission.csv: empty template with columns id, class, segmentation.
- train/: folders per case/day containing PNG slices. Filenames encode width, height, and pixel spacing.
- Pixel thickness (Z) is 3mm.
Columns:
- id: unique object ID
- class: predicted class
- segmentation: RLE-encoded mask
"""

# —— Template Config File —— 
template_config_file = """# config.yaml

project_name: "gi_tract_segmentation"
seed: 123
device: "cuda:0"
base_dir: "."
output_dir: ${base_dir}/output
fold: 0

data:
  raw_data_dir: "../uw-madison-gi-tract-image-segmentation"
  train_csv: ${data.raw_data_dir}/train.csv
  use_preprocessed_3d: True
  preprocessed_dir: ${base_dir}/preprocessed_data/fold_${fold}
  num_classes: 3
  class_names: ["large_bowel", "small_bowel", "stomach"]

preprocessing_3d:
  load:
    - name: LoadImaged
      params: { keys: ["image", "mask"], image_only: false }
  channel:
    - name: EnsureChannelFirstd
      params: { keys: ["image", "mask"] }
  orientation_spacing:
    - name: Spacingd
      params: { keys: ["image", "mask"], pixdim: [1.0, 1.0, 1.0], mode: ["bilinear", "nearest"] }
    - name: Orientationd
      params: { keys: ["image", "mask"], axcodes: "RAS" }
  intensity:
    - name: ScaleIntensityRanged
      params: { keys: ["image"], a_min: 0.0, a_max: 65535.0, b_min: 0.0, b_max: 1.0, clip: True }

augmentation:
  spatial:
    - name: RandSpatialCropd
      params: { keys: ["image", "mask"], roi_size: [160, 160, 80], random_size: false }
    - name: RandFlipd
      params: { keys: ["image", "mask"], prob: 0.5, spatial_axis: [0] }
    - name: RandAffined
      params:
        keys: ["image", "mask"]
        prob: 0.5
        rotate_range: [0.26, 0.26, 0]
        translate_range: [10, 10, 5]
        scale_range: [0.1, 0.1, 0.1]
        mode: ["bilinear", "nearest"]
        padding_mode: "reflection"

validation_transforms:
  intensity:
    - name: ScaleIntensityRanged
      params: { keys: ["image"], a_min: 0.0, a_max: 65535.0, b_min: 0.0, b_max: 1.0, clip: True }

model:
  name: "UNet"
  params:
    spatial_dims: 3
    in_channels: 1
    out_channels: 3
    channels: [32, 64, 128, 256, 512]
    strides: [2, 2, 2, 2]
    kernel_size: 3
    up_kernel_size: 3
    num_res_units: 2
    act: "PRELU"
    norm: "BATCH"
    dropout: 0.2
    bias: True

training:
  epochs: 500
  batch_size: 2
  optimizer:
    name: "Adam"
    params:
      lr: 0.0001
      weight_decay: 0.000001
  loss_function:
    name: "DiceCELoss"
    params:
      to_onehot_y: False
      sigmoid: True
      include_background: True
  lr_scheduler:
    name: "CosineAnnealingWarmRestarts"
    params:
      T_0: 100
      T_mult: 1
      eta_min: ${training.optimizer.params.lr} * 0.01
  gradient_accumulation_steps: 1
  mixed_precision: True
  run_validation: True
  validation_interval: 5

inference:
  checkpoint_path: ${output_dir}/fold_${fold}/checkpoints/best_model.pth
  mode: "test"
  batch_size: 1
  roi_size: [160, 160, 80]
  overlap: 0.5
  run_tta: True
  submission_filename: "submission.csv"

postprocessing:
  activation: "sigmoid"
  threshold: 0.5
  mask_transforms:
    - name: KeepLargestConnectedComponentd
      params: { keys: ["pred"], applied_labels: [0, 1, 2], is_onehot: False }

evaluation:
  metrics: ["DiceMetric", "HausdorffScore"]
  hausdorff_percentile: 95
  include_background: False
  reduction: "mean_batch"
  score_weights:
    dice: 0.4
    hausdorff: 0.6
"""

# —— Combined Prompt —— 
prompt = f"""
Task Description:
{task_description}

Evaluation Criteria:
{evaluation}

Dataset Details:
{dataset_description}

Please generate a valid `config.yaml` for a MONAI training pipeline that follows this template:

{template_config_file}

Your output should be a complete YAML configuration with appropriate values filled in to optimize segmentation performance.
"""

if __name__ == "__main__":
    print(prompt)
