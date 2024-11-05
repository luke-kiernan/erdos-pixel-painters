import torchvision.transforms as transforms
import warnings
from kornia import color
from math import pi

transform_basic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #Resizing to 32x32
    #transforms.Resize((32,32))
])
#Normalize to ImageNet values 
transform_Img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #Resizing to 256x256
    #transforms.Resize((256,256))
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
    #transforms.Resize((256,256))
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
    transform_in = None
    if "IN_COLORMAP" in CONFIG.keys():
        if CONFIG["IN_COLORMAP"] == "GRAY":
            transform_in = transform_basic_grayscale
        elif CONFIG["IN_COLORMAP"] == "GRAY_IMG":
            transform_in = transform_Img_grayscale
        else:
            warnings.warn("IN_COLORMAP not recognized. Using default grayscale")
            transform_in = transform_basic_grayscale
    else:
        warnings.warn("IN_COLORMAP not specified. Using default grayscale")
        transform_in = transform_basic_grayscale

    if "IMG_RESIZE" in CONFIG.keys():
        #if it is int, resize to that size
        if type(CONFIG["IMG_RESIZE"]) == int:
            transform_in.transforms.append(transforms.Resize((CONFIG["IMG_RESIZE"],CONFIG["IMG_RESIZE"])))
        elif type(CONFIG["IMG_RESIZE"]) == tuple:
            transform_in.transforms.append(transforms.Resize(CONFIG["IMG_RESIZE"]))
        else :
            warnings.warn("IMG_RESIZE was not recognized. Not resizing input transform.")
    return transform_in

def getTransform_out(CONFIG):
    transform_out = None
    if "OUT_COLORMAP" in CONFIG.keys():
        if CONFIG["OUT_COLORMAP"] == "RGB":
            transform_out =  transform_basic
        elif CONFIG["OUT_COLORMAP"] == "RGB_IMG":
            transform_out =  transform_Img
        elif CONFIG["OUT_COLORMAP"].lower() in ['lab', 'luv', 'xyz', 'hsv']:
            outType = CONFIG["OUT_COLORMAP"].lower()
            # using "Colorful Image Colorization" moralization for LAB and LUV
            # HSV is [0, 2*pi]x[0,1]^2. These ranges are specific to Kornia: openCV uses H in [0,180]
            #                           Also, there is no "middle" H bc it's a circle.
            mean, stddev = {
                'lab':((50,0,0),(100,110,110)),
                'luv':((50,0,0),(100,100,100)),
                'xyz':((0.5,0.5,0.5),(0.5,0.5,0.5)),
                'hsv':((pi,0.5,0.5),(2*pi,0.5,0.5))
            }[outType]
            if outType == 'hsv':
                warnings.warn("Normalization and distance for HSV may be incorrect on the H coordinate.",
                        "Treating cylindrical space like a product of intervals.")
            transform_out =  transforms.Compose([transforms.ToTensor(),
                ConvertColorTransform('rgb', outType),
                transforms.Normalize(mean, stddev)])
        else: 
            warnings.warn("OUT_COLORMAP not recognized. Using default RGB")
            transform_out =  transform_basic
    else:
        warnings.warn("OUT_COLORMAP not specified. Using default RGB")
        transform_out =  transform_basic

    if "IMG_RESIZE" in CONFIG.keys():
        #if it is int, resize to that size
        if type(CONFIG["IMG_RESIZE"]) == int:
            transform_out.transforms.append(transforms.Resize((CONFIG["IMG_RESIZE"],CONFIG["IMG_RESIZE"])))
        elif type(CONFIG["IMG_RESIZE"]) == tuple:
            transform_out.transforms.append(transforms.Resize(CONFIG["IMG_RESIZE"]))
        else :
            warnings.warn("IMG_RESIZE was not recognized. Not resizing output transform.")
    return transform_out

    