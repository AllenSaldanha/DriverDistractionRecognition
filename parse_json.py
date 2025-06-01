#.py file to parse through the json and extract features of a particular frame

level_names = {
    0: "Occlusion",
    1: "Gaze on Road",
    2: "Driver is Talking",
    3: "Hands using Wheel",
    4: "Hand on Gear",
    5: "Objects in Scene",
    6: "Driver Actions"
}

import json

def load_json_file():
    with open("dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json","r") as file:
        data = json.load(file)

    json_text = data["openlabel"]
    return json_text


def extract_information(frame_number,json_text_extracted_objects):
    list_of_objects = []
    for obj_id, obj_info in json_text_extracted_objects.items():
        obj_type = obj_info.get("type","Unknown")
        if check_interval(frame_number,obj_info):
            #print("Id: "+obj_id+ " Type: "+obj_type)
            list_of_objects.append(obj_type)
    return list_of_objects

def check_interval(frame_number, obj_info):
    obj_interval = obj_info.get("frame_intervals","Unknown")
    for interval in obj_interval:
        if interval["frame_start"]<=frame_number and interval["frame_end"]>=frame_number:
            return True
        
    return False
    


json_text_extracted = load_json_file()
frame_number = int(input("Enter the frame whose details you want to check: "))

#level 5
print("Level 5: Objects in Scene")
print(extract_information(frame_number, json_text_extracted["objects"]))

#level 6
print("Level 6: Driver actions")
print(extract_information(frame_number, json_text_extracted["actions"]))

