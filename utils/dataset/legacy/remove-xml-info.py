import os
import xml.etree.ElementTree as ET

XML_DIR = "$HOME/data/bottle/maid"


def main():
    xml_dir = os.path.expandvars(os.path.expanduser(XML_DIR))

    for file in os.listdir(xml_dir):
        file_name = os.path.splitext(file)[0]

        if not file.endswith(".xml"):
            continue

        with open(os.path.join(xml_dir, file), "r") as f:
            tree = ET.parse(f)

        root = tree.getroot()
        root.find("folder").text = "maid"
        root.find("filename").text = file_name + ".jpg"
        root.find("path").text = os.path.join("~/data/bottle/cola", file_name + ".jpg")
        tree.write(os.path.join(xml_dir, file))


if __name__ == "__main__":
    main()
