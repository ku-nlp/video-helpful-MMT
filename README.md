Video-Helpful Multimodal Machine Translation
==========

__EVA__ is a dataset that contains 852k Japanese-English parallel subtitle pairs, 520k Chinese-English parallel subtitle pairs, and corresponding video clips collected from movies and TV episodes: 
  - Extensive training set
  - Video-helpful evaluation set in which subtitles are ambiguous, and videos can help disambiguate the source subtitles
    
Examples:
-------------------
SRC: きれいだ

REF: You look beautiful

NMT: Beautiful

VMT: You look beautiful

![opus_1224479_305](https://github.com/ku-nlp/video-helpful-MMT/assets/38623473/d5b849c4-6535-41ec-b32b-6a882f40a621)


Splits:
-------------------

| Split      | Train     | Validation     | test     |
| ---------- | :-----------:  | :-----------: | :-----------: |
|#sample (Ja-En) | 848,164 | 2,138 | 2,138 |
|#sample (Zh-En) | 516,733 | 1,470 | 1,470 |
|video-helpful | | ✓ | ✓ |

Usage:
-------------------
You can read json files to find the mapping from videos to parallel subtitle pairs.

Json Files Structure:
-------------------  
```
"train"/"test"/"val": {
	video_file_name: {  
    		{ "ja": Japanese_subtitle },  
    		{ "en": English_subtitle }  
	}
}
```

Note:
-------------------
Please, note that by downloading the dataset, you agree to the following conditions:
  - Do not re-distribute the dataset without our permission.
  - The dataset can only be used for research purposes. Any other use is explicitly prohibited.
<!---  - We do not officially distribute the video clips. Do not disclose this anywhere, even in your paper.--->

<!--
Downloadable Features:
-------------------
If you are interested in the video features of VISA, you can download them from the following links:
  - The I3D Features of VISA: http://lotus.kuee.kyoto-u.ac.jp/~yihang/dataset/VISA_i3d.zip
  - The RCNN Features of VISA: http://lotus.kuee.kyoto-u.ac.jp/~yihang/dataset/VISA_rcnn.zip
-->

Citation:
-------------------
If you find this dataset helpful, please cite our publication "[Video-Helpful Multimodal Machine Translation](https://aclanthology.org/2023.emnlp-main.260/)":  

```
@inproceedings{li-etal-2023-video,
    title = "Video-Helpful Multimodal Machine Translation",
    author = "Li, Yihang  and
      Shimizu, Shuichiro  and
      Chu, Chenhui  and
      Kurohashi, Sadao  and
      Li, Wei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.260",
    doi = "10.18653/v1/2023.emnlp-main.260",
    pages = "4281--4299",
    abstract = "Existing multimodal machine translation (MMT) datasets consist of images and video captions or instructional video subtitles, which rarely contain linguistic ambiguity, making visual information ineffective in generating appropriate translations. Recent work has constructed an ambiguous subtitles dataset to alleviate this problem but is still limited to the problem that videos do not necessarily contribute to disambiguation. We introduce EVA (Extensive training set and Video-helpful evaluation set for Ambiguous subtitles translation), an MMT dataset containing 852k Japanese-English parallel subtitle pairs, 520k Chinese-English parallel subtitle pairs, and corresponding video clips collected from movies and TV episodes. In addition to the extensive training set, EVA contains a video-helpful evaluation set in which subtitles are ambiguous, and videos are guaranteed helpful for disambiguation. Furthermore, we propose SAFA, an MMT model based on the Selective Attention model with two novel methods: Frame attention loss and Ambiguity augmentation, aiming to use videos in EVA for disambiguation fully. Experiments on EVA show that visual information and the proposed methods can boost translation performance, and our model performs significantly better than existing MMT models.",
}
```

Contact:
-------------------
If you have any questions about this dataset, please contact l763541405@gamil.com.

License:
-------------------
[GNU General Public License v3.0](LICENSE)

