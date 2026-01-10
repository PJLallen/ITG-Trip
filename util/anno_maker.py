import os
import json
import logging
from termcolor import colored


def get_json_data(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    return data


class anno_formatter:
    """_summary_"""

    def __init__(
        self,
        cholec_dir: str,
        output_dir_name: str,
        detr_bbox_dir: str,
        files_count_hinter: int = 40,
    ) -> None:
        self.cholec_dir = cholec_dir

        cholec_anno_dir = os.path.join(cholec_dir, "debug_labels")
        self.cholec_anno_dir = cholec_anno_dir
        self.files_count_hinter = files_count_hinter

        self.output_dir_name = output_dir_name
        self.output_dir = os.path.join(cholec_dir, output_dir_name)

        self.detr_bbox_dir = detr_bbox_dir

    # typing hinting for list needs Python 3.10+
    def hidden_remover(self, input_list: list):
        """In case of .DS_Store and other hidden files

        Args:
            input_list (list): List of files

        Returns:
            [list]: List after removing hidden files
        """
        remove_list = []
        for i in range(len(input_list)):
            if input_list[i][0] == ".":
                remove_list.append(i)
        for i in remove_list:
            input_list.pop(i)
        if len(input_list) != self.files_count_hinter:
            logging.error("Mismatch between the actual and desired number of files!")
        return input_list

    def get_anno_list(self):
        self.anno_list = os.listdir(self.cholec_anno_dir)

        _ = self.hidden_remover(self.anno_list)

        anno_path = []
        for i in range(n := len(self.anno_list)):
            anno_path.append(os.path.join(self.cholec_anno_dir, self.anno_list[i]))
        self.anno_path = anno_path

    def get_output_list(self):
        self.output_anno_list = []
        if os.path.isdir(
            output_path := (os.path.join(self.cholec_dir, self.output_dir_name))
        ):
            logging.warning(
                f"Output directory {output_path} already exists, will overwrite!"
            )
        else:
            os.mkdir(output_path)
        for i in range(n := len(self.anno_list)):
            self.output_anno_list.append(os.path.join(output_path, self.anno_list[i]))

    def anno_writer(self, output_anno_path: str, vid_index: str):
        pre_data = get_json_data(os.path.join(self.cholec_anno_dir, vid_index))
        bbox_data = get_json_data(os.path.join(self.detr_bbox_dir, vid_index))
        processed_data = []
        for key, value in pre_data["annotations"].items():
            ###
            image_id = int(key)
            image_file_name = key.rjust(6, "0") + ".png"

            hoi_annotation = []
            fake_object_bbox = []
            duplicate_fliter = []
            for i in value:
                if i[7] == -1 or i[0] == -1:
                    logging.warning(
                        f"Empty triplet or verb for {image_file_name} in {vid_index}"
                    )
                    break
                else:
                    hoi_annotation.append(
                        {
                            "subject_id": -1,
                            "object_id": 0,
                            "category_id": i[7],  # Verb
                            "hoi_category_id": i[0],  # Triplet
                        }
                    )
                    fake_object_bbox.append(
                        {"bbox": [-1, -1, -1, -1], "category_id": int(i[8] + 6)}
                    )
                    duplicate_fliter.append(i[1])  # Instrument

            bbox = bbox_data["pred_boxes"][image_file_name]

            bbox_duplicate_fliter = []
            for i in bbox:
                bbox_duplicate_fliter.append(i[0])

            if len(bbox_duplicate_fliter) != len(set(bbox_duplicate_fliter)):
                logging.warning(
                    f"Duplicated bounding boxes Found for {image_file_name} in {vid_index}, will skip this image."
                )
                continue

            set_instrument = set(duplicate_fliter)

            if not hoi_annotation:
                continue

            elif len(set_instrument) != len(duplicate_fliter):
                logging.warning(
                    f"Duplicated Instruments Found for {image_file_name} in {vid_index}, will skip this image."
                )
                continue

            elif len(bbox) != len(hoi_annotation):
                logging.warning(
                    f"Number of bounding boxes and Cholec annotations mismatch for {image_file_name} in {vid_index}, will skip this image."
                )
                continue

            elif not bbox:
                logging.warning(
                    f"Empty bbox for {image_file_name} in {vid_index}, will skip this image."
                )
                continue

            else:
                bbox_anno = []
                for i in range(len_bbox := (len(bbox))):
                    bbox_anno.append(
                        {
                            "bbox": bbox[i][2:6],
                            "category_id": bbox[i][0],
                        }
                    )

            for i in range(len(bbox_anno)):
                for j in range(len(hoi_annotation)):
                    if bbox_anno[i]["category_id"] == value[j][1]:
                        hoi_annotation[j]["subject_id"] = i

            flag = False

            for i in range(len(hoi_annotation)):
                if hoi_annotation[i]["subject_id"] == -1:
                    flag = True

            if flag:
                logging.warning(
                    f"Empty subject_id for {image_file_name} in {vid_index}, will skip this image."
                )
                continue

            bbox_anno = bbox_anno + fake_object_bbox
            # if len_bbox > 1:
            #     logging.warning(
            #         f"Multiple bbox for {image_file_name} in {vid_index}"
            #     )

            for i in range(len(hoi_annotation)):
                hoi_annotation[i]["object_id"] = (
                    hoi_annotation[i]["object_id"] + len_bbox + i
                )

            ###
            processed_data.append(
                {
                    "file_name": image_file_name,
                    "img_id": image_id,
                    "annotations": bbox_anno,
                    "hoi_annotation": hoi_annotation,
                }
            )
        json.dump(processed_data, open(output_anno_path, "w"))
        pass

    def anno_meger(self):
        for i in range(self.files_count_hinter):
            writer_path = self.output_anno_list[i]
            vid_index = os.path.basename(writer_path)
            self.anno_writer(writer_path, vid_index=vid_index)

    def formatt_forward(self):
        self.get_anno_list()
        self.get_output_list()
        self.anno_meger()


if __name__ == "__main__":
    cholec_anno_dir = "/Volumes/Storage/CholecT50"
    output_dir_name = "formatted_labels"
    detr_bbox_dir = "/Volumes/Storage/CholecT50/generated_bbox_labels"

    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    logging.basicConfig(
        filename=os.path.join(cholec_anno_dir, "formatter.log"),
        filemode="w",
        encoding="utf-8",
        level=logging.DEBUG,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    formatter = anno_formatter(cholec_anno_dir, output_dir_name, detr_bbox_dir)
    formatter.formatt_forward()

    # Debugger
    pass
