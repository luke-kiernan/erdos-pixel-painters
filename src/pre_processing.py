import torchvision.transforms as transforms
import warnings
from kornia import color

transform_basic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #Resizing to 32x32
    transforms.Resize((32,32))
])
#Normalize to ImageNet values 
transform_Img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #Resizing to 256x256
    transforms.Resize((256,256))
])



transform_basic_grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_Img_grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    #Resizing to 256x256
    transforms.Resize((256,256))
])

class ConvertColorTransform(object):
    def __init__(self, fromspace, tospace):
        def validate_space(space):
            if space.lower() not in ['rgb', 'lab', 'gray', 'luv', 'xyz', 'hsv']:
                warnings.warn(f"colorspace {space} not recognized, defaulting to rgb")
                return 'Rgb'
            else:
                return space.lower().capitalize()
        fromspace = validate_space(fromspace)
        tospace = validate_space(tospace)
        if hasattr(color, f"{fromspace}To{tospace}"):
            self.transform = getattr(color, f"{fromspace}To{tospace}")
        else:
            self.transform = None
            warnings.warn("Compatible color conversion not found.")
    def __call__(self, arr):
        if self.transform is None:
            return arr
        return self.transform()(arr)

def getTransform_in(CONFIG):
    if "IN_COLORMAP" in CONFIG.keys():
        if CONFIG["IN_COLORMAP"] == "GRAY":
            #Resizing to 32x32
            return transform_basic_grayscale
        elif CONFIG["IN_COLORMAP"] == "GRAY_IMG":
            #Resizing to 256x256
            return transform_Img_grayscale
        else:
            warnings.warn("IN_COLORMAP not recognized. Using default grayscale")
            return transform_basic_grayscale
    else:
        warnings.warn("IN_COLORMAP not specified. Using default grayscale")
        return transform_basic_grayscale

def getTransform_out(CONFIG):
    if "OUT_COLORMAP" in CONFIG.keys():
        if CONFIG["OUT_COLORMAP"] == "RGB":
            return transform_basic
        elif CONFIG["OUT_COLORMAP"] == "RGB_IMG":
            return transform_Img
        elif CONFIG["OUT_COLORMAP"].lower() in ['lab', 'luv', 'xyz', 'hsv']:
            outType = CONFIG["OUT_COLORMAP"].lower()
            return transforms.Compose([transforms.ToTensor(),
                ConvertColorTransform('rgb', outType),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else: 
            warnings.warn("OUT_COLORMAP not recognized. Using default RGB")
            return transform_basic
    else:
        warnings.warn("OUT_COLORMAP not specified. Using default RGB")
        return transform_basic

    