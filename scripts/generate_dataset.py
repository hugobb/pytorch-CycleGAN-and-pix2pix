from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from torchvision.transforms import ToPILImage
import os 
from tqdm import tqdm
import copy


def main(opt):
    to_pil_image = ToPILImage()
    dataset = create_dataset(opt)
    opt.name = "%s/%s2%s" % (opt.dataset, dataset.environement_names[opt.A], dataset.environement_names[opt.B])
    model = create_model(opt) 
    model.setup(opt) 
    model.eval()
    for i, data in tqdm(enumerate(dataset), total=len(dataset.dataloader)):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()["fake_B"]
        img_path = model.get_image_paths()
        for j, path in enumerate(img_path):
            img = to_pil_image(visuals[j])
            path = path.replace("%s/%s" % (opt.dataset, dataset.A_name), "%s_augmented/%s" % (opt.dataset, dataset.B_name))
            root, ext = os.path.splitext(path)
            path = root + "_%s"%dataset.A_name + ext
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    dataset = create_dataset(opt)
    for i, A in enumerate(dataset.environement_names):
        for j, B in enumerate(dataset.environement_names):
            if A == B:
                continue
            opt = copy.deepcopy(opt)
            opt.A = i
            opt.B = j
            opt.name = "%s/%s2%s"%(opt.dataset, A, B)
            main(opt)
