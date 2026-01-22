import torch
from .Field import Fields, Parameters
from .boundary import BoundaryCondition, normalize_bcs
import inspect


class PDEModel:
    def __init__(self, shape, device, batchsize, bcs=None):
        self.shape = shape
        self.device = device
        self.batchsize = batchsize

        self.fields = Fields(shape = shape, device = device, batchsize = self.batchsize, bcs=bcs)
        self.parameters = Parameters()#batchsize = self.batchsize)
        self.num_fields = 0
        self.dyn_fields = []
        self.stat_fields = []
        self.nlmodel = None
        self.static_model = None


    def add_dynamic_field(self, name, init, L_hat):
        if not isinstance(init, torch.Tensor):
            raise TypeError(f"Initial value for field '{name}' must be a torch.Tensor, got {type(init)}")
        
        # Accepts init with or without batch dimension
        if init.shape == self.shape:
            init = init.unsqueeze(0).repeat(self.batchsize, *[1]*len(self.shape))
        elif init.shape == (self.batchsize, *self.shape):
            pass
        else:
            raise ValueError(f"Initial value for field '{name}' must have shape {self.shape} or {(self.batchsize, *self.shape)}, got {init.shape}")
        
        # Accept L_hat with or without batch dimension; detailed shape validation happens in build.
        if L_hat.dim() == len(self.shape):
            L_hat = L_hat.unsqueeze(0).repeat(self.batchsize, *[1] * len(self.shape))
        elif L_hat.dim() == len(self.shape) + 1:
            if L_hat.shape[0] != self.batchsize:
                raise ValueError(
                    f"L_hat for field '{name}' must have batch dimension {self.batchsize}, got {L_hat.shape[0]}"
                )
        else:
            raise ValueError(
                f"L_hat for field '{name}' must have {len(self.shape)} or {len(self.shape)+1} dims, got {L_hat.dim()}"
            )

        self.dyn_fields.append([name, init, L_hat])

    def add_static_field(self, name):
        self.stat_fields.append([name])  

    def set_nonlinear_model(self, model):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Nonlinear model must be a torch.nn.Module")
        self.nlmodel = model

    def set_static_compute_model(self, model):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Static compute model must be a torch.nn.Module")
        self.static_model = model

    def build(self):
        if not self.dyn_fields:
            raise ValueError("No dynamic fields added. Add at least one dynamic field.")
        
        # Check for duplicate names
        all_names = [entry[0] for entry in self.dyn_fields + self.stat_fields]
        if len(all_names) != len(set(all_names)):
            raise ValueError("Duplicate field names detected")

        count = 0
        inits = {}
        L_hats = {}
        for entry in self.dyn_fields:
            self.fields.name_to_idx[entry[0]] = count 
            count += 1
            inits[entry[0]] = entry[1]
            L_hats[entry[0]] = entry[2]
        
        self.fields.dyn_count = count
 
        for entry in self.stat_fields:
            self.fields.name_to_idx[entry[0]] = count 
            count += 1
            inits[entry[0]] = torch.zeros(self.batchsize, *self.shape)
        self.fields.stat_count = count - self.fields.dyn_count
        self.fields.finalize_field_bcs(all_names)
        self.fields.dyn_names = [entry[0] for entry in self.dyn_fields]
        self.fields.stat_names = [entry[0] for entry in self.stat_fields]

        self.fields.spatial = {name: tensor.to(self.device) for name, tensor in inits.items()}
        self.fields.spectral = self.fields.fftn()
        self.fields.L_hat = {name: tensor.to(self.device) for name, tensor in L_hats.items()}
        self._validate_L_hat_shapes()

        if self.nlmodel is None:
            self.nlmodel = ZeroModel()
        else:
            # Validate nonlinear model
            try:
                sig = inspect.signature(self.nlmodel.forward)
                if len(sig.parameters) != 2:  # fields, parameters
                    raise TypeError("Nonlinear model's forward method must accept two input parameters: fields and parameters")
                test_output = self.nlmodel(self.fields, self.parameters)
                self._validate_model_output(test_output, self.fields.dyn_names, "Nonlinear")
            except Exception as e:
                raise RuntimeError("Error occurred during nonlinear model validation.") from e


        if self.static_model is None:
            self.static_model = ZeroModel()
        else:
            # Validate static model
            try:
                sig = inspect.signature(self.static_model.forward)
                if len(sig.parameters) != 2:  # fields, parameters
                    raise TypeError("Static model's forward method must accept two input parameters: fields and parameters")
                test_output = self.static_model(self.fields, self.parameters)
                self._validate_model_output(test_output, self.fields.stat_names, "Static")
            except Exception as e:
                raise RuntimeError("Error occurred during static model validation.") from e
        

        
    
    def get_field_order(self):
        """Return the order of dynamic fields for reference when writing nonlinear models."""
        return [entry[0] for entry in self.dyn_fields]

    def compute_static(self):
        return self.static_model(self.fields, self.parameters)

    def compute_nonlinear(self):
        return self.nlmodel(self.fields, self.parameters)

    def _validate_L_hat_shapes(self):
        field_bcs = self.fields.field_bcs or {}
        for name, L_hat in self.fields.L_hat.items():
            bcs = field_bcs.get(name, normalize_bcs(None, len(self.shape)))
            expected_shape = (self.batchsize, *self._effective_spectral_shape(bcs))
            if L_hat.shape != expected_shape:
                raise ValueError(
                    f"L_hat for field '{name}' must have shape {expected_shape}, got {L_hat.shape}"
                )

    def _effective_spectral_shape(self, bcs):
        shape = []
        for N, bc in zip(self.shape, bcs):
            if bc == BoundaryCondition.DIRICHLET:
                shape.append(max(N - 2, 0))
            else:
                shape.append(N)
        return tuple(shape)

    def _validate_model_output(self, output, field_names, label):
        if not field_names:
            return
        if not isinstance(output, dict):
            raise TypeError(f"{label} model output must be dict, got {type(output)}")

        missing = [name for name in field_names if name not in output]
        if missing:
            raise ValueError(f"{label} model output missing fields: {missing}")
        for name in field_names:
            expected_shape = self.fields.spectral[name].shape
            if output[name].shape != expected_shape:
                raise ValueError(
                    f"{label} model output for '{name}' has shape {output[name].shape}, "
                    f"expected {expected_shape}"
                )


class ZeroModel(torch.nn.Module):
    def forward(self, fields, params):
        if isinstance(fields.spectral, dict):
            return {name: torch.zeros_like(fields.spectral[name]) for name in fields.name_to_idx.keys()}
        return 0
