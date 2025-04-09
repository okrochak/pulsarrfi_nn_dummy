import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import ipywidgets as widgets
from IPython.display import display

from .pipeline_methods import (
    PipelineImageToCCtoLabels,
    PipelineImageToFilterToCCtoLabels,
    PipelineImageToFilterDelGraphtoIsPulsar,
    PipelineImageToDelGraphtoIsPulsar,
)


class TunerPCA:
    """TunerPCA class is used to perform PCA on the input data and then modify the PCA components to generate new data points.
    The class is initialized with the input data and the variance threshold for PCA. The data_lists are numpy 2D arrays with the rows representing the
    samples and the the columns are the features.
    """

    def __init__(
        self,
        data_lists: np.ndarray,
        variance_threshold: float | None = None,
        normalize_features: bool = True,
        normalize_method: str = "standard", #: or minmax
    ):
        # print(f'self.data_lists_normalized: {data_lists.shape}')
        self.normalize_features = normalize_features    
        self.normalize_method = normalize_method 
        self.number_of_samples = data_lists.shape[0]   
        self.data_mins = np.ones((self.number_of_samples,1))*data_lists.min(axis=0);#print(f'DEBUG: data_mins shape: {self.data_mins.shape}')
        self.data_maxs = np.ones((self.number_of_samples,1))*data_lists.max(axis=0)
        self.data_stds = np.ones((self.number_of_samples,1))*data_lists.std(axis=0)
        self.data_means = np.ones((self.number_of_samples,1))*data_lists.mean(axis=0)

        if normalize_features:
            if self.normalize_method == "standard":
                data_lists_normalized = (data_lists - self.data_means) / (self.data_stds + 1e-6)
            elif self.normalize_method == "minmax":
                data_lists_normalized = (data_lists - self.data_mins) / (self.data_maxs - self.data_mins + 1e-6)
            else:
                raise ValueError("Invalid normalization method")
            #data_lists_normalized = (data_lists - np.vstack((self.data_means,self.data_means))) / (np.vstack((self.data_stds,self.data_stds)) + 1e-6)    
        else:
            data_lists_normalized = data_lists

        self.data_lists_normalized = data_lists_normalized
        #print(f'DEBUG: data_lists_normalized shape: {data_lists_normalized.shape}')
        self.perform_pca(variance_threshold)
        self.scaled_pca_factors = self.transformed_data.mean(axis=0)
        self.scaled_pca_factors_range = [
            (self.transformed_data[:, x].min(), self.transformed_data[:, x].max())
            for x in range(self.transformed_data.shape[1])
        ]
        self.eigenvectors = self.pca_engine.components_
        self.eigenvalues = self.pca_engine.explained_variance_ratio_

    def __call__(self):
        return self.perform_inverse_pca()

    def __repr__(self):
        return f"\nTunerPCA with components: {len(self.pca_engine.explained_variance_ratio_)} \n variance {self.pca_engine.explained_variance_ratio_} \n and eigenvectors: {self.pca_engine.components_} \n and average data in PCA space is: {self.scaled_pca_factors}"

    def perform_pca(self, variance_threshold: float | None = None):
        self.pca_engine = PCA(variance_threshold)
        self.transformed_data = self.pca_engine.fit_transform(
            self.data_lists_normalized
        )

    def modify_pca_components(self, scaled_pca_factors: np.ndarray):
        self.scaled_pca_factors = scaled_pca_factors

    def perform_inverse_pca(self):
        #print(f'DEBUG: scaled_pca_factors shape: {self.scaled_pca_factors.shape}')
        mixed_data_lists = self.pca_engine.inverse_transform(self.scaled_pca_factors)
        #print(f'DEBUG: mixed_data_lists shape: {mixed_data_lists.shape}')
        if self.normalize_features:
            if self.normalize_method == "standard":
                mixed_data_lists = mixed_data_lists * (self.data_stds[0,:] + 1e-6) + self.data_means[0,:]
            elif self.normalize_method == "minmax":
                mixed_data_lists = mixed_data_lists * (self.data_maxs[0,:] - self.data_mins[0,:] + 1e-6) + self.data_mins[0,:]
            else:
                raise ValueError("Invalid normalization method")
        return mixed_data_lists

    def plot_variance_distribution(self):
        plt.plot(self.pca_engine.explained_variance_ratio_)
        plt.xlabel("Principal Components")
        plt.ylabel("Variance")
        plt.title("Variance Distribution")
        plt.show()


class TunableParameterExtractor:
    
    @staticmethod
    def pull_parameters(extract_from:nn.Module
            | PipelineImageToCCtoLabels
            | PipelineImageToFilterToCCtoLabels
            | PipelineImageToFilterDelGraphtoIsPulsar
            | PipelineImageToDelGraphtoIsPulsar):
        if isinstance(extract_from, nn.Module):
            tunable_parameters_flattened, parameter_lengths = (
                TunableParameterExtractor.pull_learnable_params_from_torch_module(extract_from)
            )

        elif isinstance(
            extract_from,
            (
                PipelineImageToCCtoLabels,
                PipelineImageToFilterToCCtoLabels,
                PipelineImageToFilterDelGraphtoIsPulsar,
                PipelineImageToDelGraphtoIsPulsar,
            ),
        ):
            tunable_parameters_flattened, parameter_lengths = (
                TunableParameterExtractor.pull_learnable_params_from_pipeline(pipeline=extract_from)
            )
        else:
            raise ValueError("Invalid extract_from input type")

        return tunable_parameters_flattened, parameter_lengths

    @staticmethod
    def push_parameters(extracted_from, tunable_parameters_flattened, parameter_lengths):
        if isinstance(extracted_from, nn.Module):
            TunableParameterExtractor.push_learnable_params_to_torch_module(
                extracted_from, tunable_parameters_flattened
            )
        elif isinstance(
            extracted_from,
            (
                PipelineImageToCCtoLabels,
                PipelineImageToFilterToCCtoLabels,
                PipelineImageToFilterDelGraphtoIsPulsar,
                PipelineImageToDelGraphtoIsPulsar,
            ),
        ):
            TunableParameterExtractor.push_learnable_params_to_pipeline(
                extracted_from, tunable_parameters_flattened, parameter_lengths
            )
        else:
            raise ValueError("Invalid extract_from input type")

    @staticmethod
    def pull_learnable_params_from_torch_module(neural_network: nn.Module):
        params = []
        param_lengths = []
        for param in neural_network.parameters():
            params.append(
                param.data.view(-1).cpu().numpy()
            )  # Flatten and convert to NumPy
        params = np.concatenate(params)
        param_lengths = len(params)
        return params, param_lengths

    @staticmethod
    def push_learnable_params_to_torch_module(
        neural_network: nn.Module, learnable_params: np.ndarray
    ):
        idx = 0
        for param in neural_network.parameters():
            param.data = torch.from_numpy(
                learnable_params[idx : idx + param.numel()]
            ).view(param.size()).float()
            idx += param.numel()
        return neural_network

    @staticmethod
    def pull_learnable_params_from_pipeline(        
        pipeline: (
            PipelineImageToCCtoLabels
            | PipelineImageToFilterToCCtoLabels
            | PipelineImageToFilterDelGraphtoIsPulsar
            | PipelineImageToDelGraphtoIsPulsar
        ),
    ):
        param_dict = pipeline.__dict__
        params = []
        param_lengths = []
        for param in param_dict.values():
            if isinstance(param, nn.Module):
                params_current, param_lengths_current = (
                    TunableParameterExtractor.pull_learnable_params_from_torch_module(param)
                )
                params.append(params_current)
                param_lengths.append(param_lengths_current)
        return np.concatenate(params), param_lengths

    @staticmethod
    def push_learnable_params_to_pipeline(        
        pipeline: (
            PipelineImageToCCtoLabels
            | PipelineImageToFilterToCCtoLabels
            | PipelineImageToFilterDelGraphtoIsPulsar
            | PipelineImageToDelGraphtoIsPulsar
        ),
        learnable_params: np.ndarray,
        param_lengths: list,
    ):
        param_dict = pipeline.__dict__
        idx = 0
        counter = 0
        for param in param_dict.values():
            if isinstance(param, nn.Module):
                idx_end = idx + param_lengths[counter]
                TunableParameterExtractor.push_learnable_params_to_torch_module(
                    param, learnable_params[idx:idx_end]
                )
                idx += param_lengths[counter]
                counter += 1
        return pipeline
    
class Tuner:
    def __init__(self,
                 sample_of_objects: list[nn.Module |
                    PipelineImageToCCtoLabels |
                    PipelineImageToFilterToCCtoLabels |
                    PipelineImageToFilterDelGraphtoIsPulsar |
                    PipelineImageToDelGraphtoIsPulsar],
                show_steps:bool = True,
                reset_components:bool = True
                ):
        self.sample_of_objects = sample_of_objects
        self.show_steps = show_steps
        self.reset_components = reset_components    
        self.sliders = []
        self.current_component:int = 0
        self.current_value:float = 0.5
        self.initiate_pca_modules()
        self.generate_sliders_for_tuning()
               

    def __call__(self):
        mxd_md = self.generate_mixed_model()
        return mxd_md 

    def initiate_pca_modules(self):  
        #: extract parameters from object and linearize it
        if self.show_steps:print(f'MESSAGE: Extracting parameters from object of type {type(self.sample_of_objects[0])}')
        results_of_pulled_parameters = [TunableParameterExtractor.pull_parameters(x) for x in self.sample_of_objects]
        flattened_parameters_numpy = np.array([x[0] for x in results_of_pulled_parameters])
        self.parameter_lengths = results_of_pulled_parameters[0][1]        

        #: transform the parameters to PCA space        
        if self.show_steps:print(f'MESSAGE: Transforming parameters of length {self.parameter_lengths} the flattened to PCA space')
        self.pca_engine = TunerPCA(flattened_parameters_numpy, 0.9)

        #: calculate the number of components by knowing the number of eigenvalues
        self.num_components = len(self.pca_engine.eigenvalues)
        if self.show_steps:print(f'MESSAGE: Number of components in PCA space is {self.num_components}') 
        self.avg_scaled_pca_factors = self.pca_engine.scaled_pca_factors.copy()    

    def set_component(self,component: int):
        self.current_component = int(component)
        #self.update_avg_scaling_parameters()
        #print(f'MESSAGE: Current component set to {component} and value is {self.current_value} and the scaledparams are {self.pca_engine.scaled_pca_factors}')

    def set_current_value(self,current_value: float):
        self.current_value = current_value 
        self.update_avg_scaling_parameters()
        formated_display = [f'{x:.2f}' for x in self.pca_engine.scaled_pca_factors]
        print(f'MESSAGE: Current components set to {self.current_component} and value is {self.current_value} and the scaledparams are {formated_display}')  

    def generate_sliders_for_tuning(self):
        component_slider_tool = NotebookGUITools(self.set_component)
        value_sliders_range_list = self.pca_engine.scaled_pca_factors_range
        component_slider_tool.slider(min_value=0, max_value=self.num_components-1, step=1, value=0,description='Component:')
        value_slider_tool = NotebookGUITools(self.set_current_value) 
        value_slider_tool.slider(min_value=value_sliders_range_list[self.current_component][0],max_value=value_sliders_range_list[self.current_component][1],step=0.1,value=self.pca_engine.scaled_pca_factors[self.current_component],description='Value:')

    def update_avg_scaling_parameters(self):        
        if self.reset_components:
            self.pca_engine.scaled_pca_factors = self.avg_scaled_pca_factors.copy()
        self.pca_engine.scaled_pca_factors[self.current_component] = self.current_value
          
    def generate_mixed_model(self):
        self.update_avg_scaling_parameters()
        if self.show_steps:print(f'MESSAGE: Generating new model with scaled parameters {self.pca_engine.scaled_pca_factors}')
        #: calculate the inverse pca in normalized form
        mixed_parameters = self.pca_engine.perform_inverse_pca()
        #print(f'DEBUG: mixed_parameters[0:10]: {mixed_parameters[0:10]}')
        #print(f'DEBUG: mixed_parameters shape: {mixed_parameters.shape}')   
        #: push the parameters back to the objects
        mixed_obj = deepcopy(self.sample_of_objects[0])
        param_before = TunableParameterExtractor.pull_parameters(mixed_obj)
        TunableParameterExtractor.push_parameters(mixed_obj,mixed_parameters,self.parameter_lengths)
        param_after = TunableParameterExtractor.pull_parameters(mixed_obj)
        diff = param_after[0] - param_before[0]
        #print(f'DEBUG: diff is : {diff[0:10]}')        
        return mixed_obj         
        





def example_callback(change):
    print('Slider moved to :',change.new)

class NotebookGUITools:
    def __init__(self, function: callable = None):
        self.function = function if function else example_callback  # Default to example_callback
        self.output = widgets.Output()
        
    def function_wrapper(self, change):
        with self.output:
            self.output.clear_output(wait=True)
            self.function(change.new)  # Pass entire change dict

    def slider(self, min_value: float = 0.0, max_value: float = 1.0, step: float = 0.1, value: float = 0.5,description:str='Slider:'):  
        slider = widgets.FloatSlider(
            value=value,
            min=min_value,
            max=max_value,
            step=step,
            description=description,
            continuous_update=True,  # Ensures real-time updates
            readout_format=".1f",
        )

        slider.observe(self.function_wrapper, names="value")  # Uses function_wrapper
        display(slider, self.output)  # Display slider and output widget
        return slider, self.output




    
    
           
       
    
    
        


    


if __name__ == "__main__":
    data = np.random.rand(6, 10) * 10
    print("data", data)
    tuner = TunerPCA(data, 0.9)
    print(tuner)
    print(f"mixed data is {tuner()}")
