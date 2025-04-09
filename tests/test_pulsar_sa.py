import os
import numpy as np
import pytest

def test_import():       
    try:
        from src import (
                         PrepareFreqTimeImage,
                         ImageReader,
                         ImageDataSet,
                         LabelReader,
                         LabelDataSet,
                         PipelineImageToCCtoLabels,
                         PipelineImageToDelGraphtoIsPulsar,
                         PipelineImageToFilterDelGraphtoIsPulsar,
                         PipelineImageToFilterToCCtoLabels,
                         PipelineImageToMask,
                         ImageToMaskDataset,
                         InMaskToMaskDataset,
                         TrainImageToMaskNetworkModel, 
                         TrainSignalToLabelModel,
                         UNet, FilterCNN, CNN1D,
                         Tuner
                         )
        print('MESSAGE: Imported modules from root')
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_directories():
    directories = [
    "./unet_semantic_segmentation/syn_data",
    "./unet_semantic_segmentation/syn_data/runtime/",
    "./unet_semantic_segmentation/syn_data/payloads/",
    "./unet_semantic_segmentation/syn_data/model/",    
    ]
    #print('cwd',os.getcwd())
    for directory in directories:
        if not os.path.exists(directory):
            assert False, f"Necessary directory {directory} doesn't exist. run the setup_config.py file"
        else:
            assert True

@pytest.fixture
def loader_from_payload_to_image():
    from src import PrepareFreqTimeImage
    image_preprocessing_engine = PrepareFreqTimeImage(
                                                do_rot_phase_avg=True,
                                                do_binarize=False,
                                                do_resize=True,
                                                resize_size=(128,128),
                                                )
    return image_preprocessing_engine

def __test_write_payload_to_numpy(loader_from_payload_to_image):
    """_summary_

    Args:
        loader_from_payload_to_image (_type_): _description_
    """    
    from src import ImageDataSet,ImageReader
    image_tag='test_example_0_payload_detected.json'
    image_directory='./unet_semantic_segmentation/syn_data/payloads/'
    im_set = ImageDataSet(image_tag=image_tag,image_directory=image_directory,image_reader_engine=ImageReader(do_average=True))
    numpy_image:np.ndarray = im_set[0]
    np.save('./unet_semantic_segmentation/tests/data/payload_to_numpy_image.npy',numpy_image)

def test_payload_to_image(loader_from_payload_to_image):
    from src import ImageDataSet,ImageReader 
    test_example_data = np.load('./unet_semantic_segmentation/tests/data/payload_to_numpy_image.npy') 
    image_tag='test_example_0_payload_detected.json'
    image_directory='./unet_semantic_segmentation/syn_data/payloads/'
    im_set = ImageDataSet(image_tag=image_tag,image_directory=image_directory,image_reader_engine=ImageReader(do_average=True))   
    generated_data :np.ndarray = im_set[0]
    diff_metric = (generated_data - test_example_data)**2
    diff_metric_min = diff_metric.min()
    diff_metric_max = diff_metric.max()
    if (diff_metric_min == pytest.approx(diff_metric_max)) and diff_metric_max==0:
        print(f'MESSAGE: Generated data  matched the pre-generated data by developers with diff_metric_max - diff_metric_min = {diff_metric_max - diff_metric_min} and diff_metric_max = {diff_metric_max}')
        assert True
    else:
        assert False , f'Generated data didnt match the pre-generated data by developers and deviates by {diff_metric_max - diff_metric_min}'



    



