import torch
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
            encoder_name='resnet34', 
            classes=7,
            encoder_weights="imagenet"
        )
model.load_state_dict(torch.load('/AkhmetzyanovD/projects/nztfm/main_pipeline/runs/segm_fpn_mit_b1/launch_0/weights/best_model'))
model = model.to('cpu')
model.eval()

input_tensor = torch.randn((1, 3, 1072, 1920))

torch.onnx.export(model, 
                  input_tensor, 
                 '/AkhmetzyanovD/projects/nztfm/main_pipeline/runs/segm_fpn_mit_b1/launch_0/weights/best_model.onnx',
                  input_names=['input'], 
                  output_names=['output'],
                  export_params=True,
                )

print('Model converted to ONNX format and saved to model.onnx')