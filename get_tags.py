# -*- coding: utf-8 -*-

import os
import json

def main():
    image_path = "SEM\高晶面图"
    dirs = os.listdir(image_path)
    print(dirs)
    tags = {}
    for image in dirs:
        print(image)
        if image[3] == "与":
            tags[image] = [image[:3], image[4:7]]
        else:
            tags[image] = [image[:3]]
    print(tags)
    #filename = os.path.join(image_path, 'tags.json')
    filename = 'tags.json'
    with open(filename, 'w') as file_obj:
        json.dump(tags, file_obj)

if __name__ == "__main__":
    main()
    

    