from argparse import ArgumentParser
from transformers import AutoModelForImageSegmentation, AutoProcessor, CLIPSegForImageSegmentation, SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import pipeline as transformer_pipeline
from diffusers.utils import make_image_grid
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from os.path import basename, dirname, join
from math import ceil
from random import randint
import torch
import cv2
import numpy as np
torch.set_float32_matmul_precision(['high', 'highest'][0])

class Segmenter(ABC):
    @abstractmethod
    def __init__(self): ...
                
    @property
    @abstractmethod
    def model_name(self): ...
        
    def prep_image(self, image): return image
        
    @abstractmethod
    def run_model(self, prepped_image): pass
        
    def segment(self, image): return self.run_model(self.prep_image(image))

class Birefnet(Segmenter):
    def __init__(self, device="cuda"):
        self.model = AutoModelForImageSegmentation.from_pretrained(self.hf_model, trust_remote_code=True)
        self.model.to(device)
        self.device = device
        self.model.eval()
        self.model.half()
    @property
    def hf_model(self): return "ZhengPeng7/BiRefNet"
    @property
    def model_name(self): return "BiRefNet"
    def prep_image(self, image):
        return transforms.ToTensor()(image.convert("RGB")).unsqueeze(0).to(self.device).half()
    def run_model(self, prepped_image):
        return {"background_segmentation": transforms.ToPILImage()(self.model(prepped_image)[-1].sigmoid().cpu()[0].squeeze())}

class Birefnet_HR(Birefnet):
    @property
    def hf_model(self): return "ZhengPeng7/BiRefNet_HR"
    @property
    def model_name(self): return "BiRefNet_HR"

class Segformer_B2_Clothes(Segmenter):
    def __init__(self, device="cuda"):
        self.model = transformer_pipeline(model="mattmdjaga/segformer_b2_clothes", device=device)
    @property
    def model_name(self): return "segformer_b2_clothes"

    @property
    def clothes_segments(self): return ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Scarf"]

    @property
    def fullbody_segments(self): return self.clothes_segments + ["Hat", "Hair", "Sunglasses", "Face", "Bag", "Left-shoe", "Right-shoe", "Left-leg", "Right-leg", "Left-arm", "Right-arm"]
   
    def run_model(self, prepped_image):
        segments = self.model(prepped_image)
        mask = None
        fullbody_mask = None
        for segment in segments:
            if segment['label'] in self.clothes_segments:
                if not mask: mask = segment['mask']
                else: mask.paste(segment['mask'], mask=segment['mask'])
            if segment['label'] in self.fullbody_segments:
                if not fullbody_mask: fullbody_mask = segment['mask']
                else: fullbody_mask.paste(segment['mask'], mask=segment['mask'])
        return {"background_segmentation": fullbody_mask, "clothing_segmentation": mask}

class RMBG14(Segmenter):
    def __init__(self, device="cuda"):
        self.model = transformer_pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)
    @property
    def model_name(self): return "RMBG_1.4"
    def prep_image(self, image): return image
    def run_model(self, prepped_image): return {"background_segmentation": self.model(prepped_image, return_mask = True)}

class CLIPSeg_RD64_Refined(Segmenter):
    def __init__(self, device=None):
        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    @property
    def model_name(self): return "CLIPSeg_RD64_refined"
    def prep_image(self, image):
        proc = self.processor(text=["clothing"], images=[image], padding=True, return_tensors="pt")
        return [proc, image.size]
    def run_model(self, prepped_image):
        proc = prepped_image[0]
        size = prepped_image[1]
        out = self.model(**proc)
        pred = out.logits.unsqueeze(1)
        sigmoid = torch.sigmoid(pred[0][0])
        transform = transforms.ToPILImage()
        img = transform(sigmoid).convert('L').resize(size)
        #img = img.point(lambda p: 255 if p > 20 else 0).convert('1') # uncomment to enable binary threshold
        return {"clothing_segmentation": img}

class DeeplabV3_Resnet50(Segmenter):
    def __init__(self, device="cuda"):
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model.eval().to(device)
        self.device = device
        self.transform = transforms.Compose([
            #transforms.Resize((1216,832)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    @property
    def model_name(self): return "DeeplabV3_Resnet50"
    def prep_image(self, image):
        #return transforms.ToTensor()(image.convert("RGB")).unsqueeze(0).to(self.device)
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
    @property
    def random_colors(self): return np.array([[ 0, 0,  0], # background
                                       [ 18,  98, 145], # these ones are random,
                                       [207,  56, 248], # I just pasted them in so
                                       [ 90, 146, 253], # they'll stay the same
                                       [210,  18,   3],
                                       [169,  70, 199],
                                       [205,  14,   2],
                                       [179, 179, 116],
                                       [112, 201, 161],
                                       [ 90, 181, 208],
                                       [ 30, 234, 168],
                                       [135, 155, 148],
                                       [ 77,  60,   6],
                                       [172, 202, 249],
                                       [103, 103,  83],
                                       [255, 255, 255], # person
                                       [250,  66, 220],
                                       [178, 168, 116],
                                       [ 33, 104, 160],
                                       [190, 235,  47],
                                       [ 53,  17, 189]], dtype=np.uint8)
    @property
    def person_colors(self): return np.array([[ 0, 0,  0], # background
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [255, 255, 255], # person
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0],
                                       [ 0, 0,  0]], dtype=np.uint8)
    def segmentation_to_color(self, segmentation, colors): # from seg_attacks
        num_classes = 21  # for DeepLabV3 with COCO (21 classes including background)
        # colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        color_segmentation = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for label in range(num_classes):
            color_segmentation[segmentation == label] = colors[label]
        return color_segmentation
    def run_model(self, prepped_image):
        output = self.model(prepped_image)['out']
        segmentation = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        np_segmentation = self.segmentation_to_color(segmentation, self.random_colors)
        person_segmentation = self.segmentation_to_color(segmentation, self.person_colors)
        return {"semantic_segmentation": Image.fromarray(np_segmentation),
                #"np_segmentation": np_segmentation,
               "background_segmentation": Image.fromarray(person_segmentation).convert("1")}

'''
class FaceParsing(Segmenter):
    def __init__(self, device="cuda"):
        self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.model.to(device)
        self.device = device
    @property
    def model_name(self): return "FaceParsing"
    def prep_image(self, image):
        processed_img = self.image_processor(images=image, return_tensors="pt").to(self.device)
        processed_img['size'] = image.size
    def run_model(self, prepped_image):
        outputs = self.model(**prepped_image)
        resized = torch.nn.functional.interpolate(outputs.logits, size=prepped_image['size'][::-1], mode="bilinear", align_corners=False)
        labels = resized.argmax(dim=1)[0]
'''

class FaceParsing(Segmenter):
    def __init__(self, device="cuda"):
        self.model = transformer_pipeline(model="jonathandinu/face-parsing", device=device)
    @property
    def model_name(self): return "FaceParsing"

    @property
    def clothes_segments(self): return ["cloth", "skin"]

    @property
    def fullbody_segments(self): return self.clothes_segments + ["nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck"]
   
    def run_model(self, prepped_image):
        segments = self.model(prepped_image)
        mask = None
        fullbody_mask = None
        for segment in segments:
            if segment['label'] in self.clothes_segments:
                if not mask: mask = segment['mask']
                else: mask.paste(segment['mask'], mask=segment['mask'])
            if segment['label'] in self.fullbody_segments:
                if not fullbody_mask: fullbody_mask = segment['mask']
                else: fullbody_mask.paste(segment['mask'], mask=segment['mask'])
        return {"background_segmentation": fullbody_mask, "clothing_segmentation": mask}

class Maskformer_SWIN_Large_COCO(Segmenter):
    def __init__(self, device="cuda"):
        self.pipeline = transformer_pipeline(model="facebook/maskformer-swin-large-coco", device=device)

    @property
    def model_name(self): return "Maskformer_SWIN_Large_COCO"

    def run_model(self, prepped_image):
        out = self.pipeline(prepped_image)
        for mask in filter(lambda m: m['label'] == "person", out):
            return {"background_segmentation": mask['mask']}
        return {}

class SegmentAnything(Segmenter):
    def __init__(self, device="cuda"):
        self.pipeline = transformer_pipeline(model=self.hf_model, device=device)

    def run_model(self, prepped_image):
        out = self.pipeline(prepped_image)
        '''
        new_img = prepped_image.copy().convert("RGB")
        for mask in out['masks']:
            mask_base = Image.fromarray(mask)
            mask_img = mask_base.convert("L")
            mask_img = ImageOps.colorize(mask_img, black=(0,0,0), white=(randint(100,255),randint(100,255),randint(100,255))).convert("RGBA")
            cropped_mask_img = Image.new("RGBA", mask_img.size, (0, 0, 0, 0))
            cropped_mask_img.paste(mask_img, mask=mask_base)
            new_img = Image.blend(new_img.convert("RGBA"), cropped_mask_img, 0.8)
        return {"segment_anything": new_img}
        '''
        return {"segment_anything_" + str(i): Image.fromarray(v) for i, v in enumerate(out['masks'])}

class SegmentAnything_VIT_Base(SegmentAnything):
    @property
    def hf_model(self): return "facebook/sam-vit-base"

    @property
    def model_name(self): return "SegmentAnything_VIT_Base"

class SegmentAnything_VIT_Large(SegmentAnything):
    @property
    def hf_model(self): return "facebook/sam-vit-large"

    @property
    def model_name(self): return "SegmentAnything_VIT_Large"

class SegmentAnything_VIT_Huge(SegmentAnything):
    @property
    def hf_model(self): return "facebook/sam-vit-huge"

    @property
    def model_name(self): return "SegmentAnything_VIT_Huge"

class SegmentAnything2_Hiera_Large(SegmentAnything):
    @property
    def hf_model(self): return "facebook/sam2-hiera-large"

    @property
    def model_name(self): return "SegmentAnything2_Hiera_Large"

SEGMENTERS = {"BiRefNet": Birefnet, "BiRefNet_HR": Birefnet_HR, "segformer_b2_clothes": Segformer_B2_Clothes,
              "RMBG_1.4": RMBG14, "CLIPSeg_RD64_refined": CLIPSeg_RD64_Refined, "DeeplabV3_Resnet50": DeeplabV3_Resnet50,
              "FaceParsing": FaceParsing, "Maskformer_SWIN_Large_COCO": Maskformer_SWIN_Large_COCO,
              "SegmentAnything_VIT_Base": SegmentAnything_VIT_Base, "SegmentAnything_VIT_Large": SegmentAnything_VIT_Large,
              "SegmentAnything_VIT_Huge": SegmentAnything_VIT_Huge,
             "SegmentAnything2_Hiera_Large": SegmentAnything2_Hiera_Large}
VALID_MASK_TYPES = ["background_segmentation", "clothing_segmentation", "semantic_segmentation"]

def main(): # from segmentation_evaluation_sensation.ipynb
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-i", nargs="+", dest="images")
    args = parser.parse_args()
    device = args.device
    segmenters = [Birefnet(device), Birefnet_HR(device),
                  Segformer_B2_Clothes(device), CLIPSeg_RD64_Refined(),
                  RMBG14(device), DeeplabV3_Resnet50(device), FaceParsing(device),
                  Maskformer_SWIN_Large_COCO(device),
                  SegmentAnything_VIT_Base(device), SegmentAnything_VIT_Large(device), SegmentAnything_VIT_Huge(device)]
    for imgpath in args.images:
        image = Image.open(imgpath).convert("RGB")
        imgs = []
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 40)
        IMGS_PER_ROW = 4
        for segmenter in segmenters:
            masks = segmenter.segment(image)
            for name, mask in masks.items():
                #if name not in VALID_MASK_TYPES: continue
                mask.save(join("masks", "mask", basename(imgpath).replace(".png", "_" + segmenter.model_name + "_" + name + ".png")))
                mask_texted = mask.copy().convert("RGB")
                draw_mask = ImageDraw.Draw(mask_texted)
                draw_mask.text((10, 10), segmenter.model_name + "\n" + name, font=font, fill=(56, 255, 255))
                imgs.append(mask_texted)
                if name != "semantic_segmentation": # can't use semantic_segmentation as b/w mask
                    cropped = Image.new("RGB", image.size, (255, 255, 255))
                    cropped.paste(image, mask=mask)
                    draw_cropped = ImageDraw.Draw(cropped)
                    draw_cropped.text((10, 10), segmenter.model_name + "\n" + name, font=font, fill=(255))
                    imgs.append(cropped)
        if len(imgs) <= IMGS_PER_ROW: rows, cols = 1, len(imgs)
        else: rows, cols = ceil(len(imgs) / IMGS_PER_ROW), IMGS_PER_ROW
        while rows*cols > len(imgs): # fill extra space with white
            imgs.append(Image.new("RGB", image.size, (255, 255, 255)))
        grid = make_image_grid(imgs, rows=rows, cols=cols)
        grid.save(join("masks", "grid", basename(imgpath).replace(".png", "_grid.png")))
        smallgrid = grid.resize((grid.width//2, grid.height//2))
        smallgrid.save(join("masks", "grid", basename(imgpath).replace(".png", "_smallgrid.png")))

if __name__ == "__main__": main()