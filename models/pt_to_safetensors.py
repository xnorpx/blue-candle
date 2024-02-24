import argparse
import os
from torch import load
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser(description="Convert .pt file to .safetensors")
    parser.add_argument("pt_file", type=str, help="Path to the .pt file")
    parser.add_argument("-s", "--safetensors", type=str, help="Path to the .safetensors file (optional)")

    args = parser.parse_args()

    pt_filename = args.pt_file
    sf_filename = args.safetensors

    if not sf_filename:
        # If safetensors filename is not provided, use the pt filename with .safetensors extension
        sf_filename = pt_filename.rsplit('.', 1)[0] + '.safetensors'

    print("Converting {} to {}".format(pt_filename, sf_filename))
    convert_file(pt_filename, sf_filename)
    print("Done, make sure to save labels in a yaml file with name: {}.yaml".format(sf_filename.rsplit('.', 1)[0]))

def convert_file(
    pt_filename: str,
    sf_filename: str,
):

    data = load(pt_filename)
    tensors = data['model'].state_dict().items()
    tensors = dict(tensors)
    tensors = {rename(k): t for k, t in tensors.items()}
    save_file(tensors, sf_filename)

def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )

def rename(name: str):
    name = name.replace("model.0.", "net.b1.0.")
    name = name.replace("model.1.", "net.b1.1.")
    name = name.replace("model.2.m.", "net.b2.0.bottleneck.")
    name = name.replace("model.2.", "net.b2.0.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.4.m.", "net.b2.2.bottleneck.")
    name = name.replace("model.4.", "net.b2.2.")
    name = name.replace("model.5.", "net.b3.0.")
    name = name.replace("model.6.m.", "net.b3.1.bottleneck.")
    name = name.replace("model.6.", "net.b3.1.")
    name = name.replace("model.7.", "net.b4.0.")
    name = name.replace("model.8.m.", "net.b4.1.bottleneck.")
    name = name.replace("model.8.", "net.b4.1.")
    name = name.replace("model.9.", "net.b5.0.")
    name = name.replace("model.12.m.", "fpn.n1.bottleneck.")
    name = name.replace("model.12.", "fpn.n1.")
    name = name.replace("model.15.m.", "fpn.n2.bottleneck.")
    name = name.replace("model.15.", "fpn.n2.")
    name = name.replace("model.16.", "fpn.n3.")
    name = name.replace("model.18.m.", "fpn.n4.bottleneck.")
    name = name.replace("model.18.", "fpn.n4.")
    name = name.replace("model.19.", "fpn.n5.")
    name = name.replace("model.21.m.", "fpn.n6.bottleneck.")
    name = name.replace("model.21.", "fpn.n6.")
    name = name.replace("model.22.", "head.")
    return name

if __name__ == "__main__":
    main()
