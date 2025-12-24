import json
import os
from pathlib import Path
import datasets


_CITATION = """
@misc{gurari2018vizwiz,
      title={VizWiz Grand Challenge: Answering Visual Questions from Blind People}, 
      author={Danna Gurari and Qing Li and Abigale J. Stangl and Anhong Guo and Chi Lin and Kristen Grauman and Jiebo Luo and Jeffrey P. Bigham},
      year={2018},
      eprint={1802.08218},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_HOMEPAGE = "https://vizwiz.org/tasks-and-datasets/vqa/"

_DESCRIPTION = """
The VizWiz-VQA dataset originates from a natural visual question answering setting where blind people 
each took an image and recorded a spoken question about it, together with 10 crowdsourced answers per 
visual question. The proposed challenge addresses the following two tasks for this dataset: predict the 
answer to a visual question and (2) predict whether a visual question cannot be answered.
"""

_LICENSE = " Creative Commons Attribution 4.0 International License."

_DATA_URL = {"train" : "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
             "test"  : "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip",
             "val"   : "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip" }

_ANNOTATION_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"

_FEATURES = datasets.Features(
    {
        "id"   : datasets.Value("int32"),
        "image": datasets.Image(),
        "filename": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "answers_original": [
            {
                "answer": datasets.Value("string"),
                "answer_confidence": datasets.Value("string"),
            }
        ],
        "answer_type": datasets.Value("string"),
        "answerable": datasets.Value("int32")
    }
)


class VizWiz(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    def _info(self):
        return datasets.DatasetInfo(
            description = _DESCRIPTION,
            features    = _FEATURES,
            homepage    = _HOMEPAGE,
            license     = _LICENSE,
            citation    = _CITATION,
        )
    
    def _split_generators(self, dl_manager):
        ann_file_train = os.path.join(dl_manager.download_and_extract(_ANNOTATION_URL), "train.json")
        ann_file_val   = os.path.join(dl_manager.download_and_extract(_ANNOTATION_URL), "val.json")
        ann_file_test  = os.path.join(dl_manager.download_and_extract(_ANNOTATION_URL), "test.json")
        image_folders = {k: Path(v) for k, v in dl_manager.download_and_extract(_DATA_URL).items()}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": ann_file_train,
                    "image_folders": image_folders,
                    "split_key": 'train'
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": ann_file_val,
                    "image_folders": image_folders,
                    "split_key": "val"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file": ann_file_test,
                    "image_folders": image_folders,
                    "split_key": "test"
                },
            ),
        ]
    
    def _generate_examples(self, annotation_file,image_folders,split_key):
        counter = 0
        annotations = json.load(open(annotation_file))
        for ann in annotations:
            if split_key in ['train','val']:
                answers = [answer["answer"] for answer in ann["answers"]]
                answers_original = ann['answers']
                answer_type = ann["answer_type"]
                answerable  = ann["answerable"]
                
            else:
                
                answers = None
                answers_original = None
                answer_type = None
                answerable  = None
                
            yield counter, {
                        "id"   : counter,
                        "image": str(image_folders[split_key]/split_key/ann['image']),
                        "filename" : ann['image'],
                        "question" : ann["question"],
                        "answers"  : answers,
                        "answers_original"  : answers_original,
                        "answer_type"       : answer_type,
                        "answerable"        : answerable
                    }
            counter += 1