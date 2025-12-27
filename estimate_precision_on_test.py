import torch
from torch.utils.data import DataLoader

from fractals_dataset import FractalImageadDataset
from training_example_02 import ParamPredictor


def estimate_precision_on_test(model, test_dataloader):
    model.eval()
    mse_loss = torch.nn.MSELoss()

    num_pictures = 0
    total_test_loss = 0.0

    with torch.no_grad():
        for image_batch in test_dataloader:
            images, coeffs = image_batch
            out_tensor = model(images)
            batch_loss = mse_loss(out_tensor, coeffs)

            total_test_loss += batch_loss.item()
            num_pictures += len(images)
        avg_test_loss = total_test_loss / num_pictures

        print(f"Number of pictures: {num_pictures}, average test loss: {avg_test_loss:.3f}")


if __name__ == "__main__":
    model = ParamPredictor()

    saved_model_path = "train_custom_02/custom_model_epoch_20.ckpt"
    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict=state_dict)

    images_folder = "dataset_preparation/fractals_2_colors"
    test_anno_fpath_1 = "dataset_preparation/fractals_2_colors/params_test_6.json"
    test_anno_fpath_2 = "dataset_preparation/fractals_2_colors/params_test_filtered_6.json"

    for test_anno_fpath in [test_anno_fpath_1, test_anno_fpath_2]:
        test_dataset = FractalImageadDataset(test_anno_fpath, images_folder)

        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        estimate_precision_on_test(model, test_dataloader)
