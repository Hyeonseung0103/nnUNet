from typing import Tuple, Union, List

from batchgenerators.utilities.file_and_folder_operations import save_json, join


def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          citation: Union[List[str], str] = None,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None,
                          reference: str = None,
                          release: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None,
                          license: str = 'Whoever converted this dataset was lazy and didn\'t look it up!',
                          converted_by: str = "Please enter your name, especially when sharing datasets with others in a common infrastructure!",
                          **kwargs):
    """
    Generates a dataset.json file in the output folder

    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
        'licence': license,
        'converted_by': converted_by
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if citation is not None:
        dataset_json['citation'] = release
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)

generate_dataset_json(output_folder = './nnUNet/',
                      channel_names = {
                            "0": "CT"
                        },
                      labels = {
                        "background": 0,
                        "head-frac": 1,
                        "skull": 2
                      },
                      num_training_cases = 2,
                      file_ending = '.nrrd')