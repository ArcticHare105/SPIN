## SPIN

Codes for "Lightweight Image Super-Resolution with Superpixel Token Interaction",

> **Lightweight Image Super-Resolution with Superpixel Token Interaction** <br>
> Aiping Zhang (Sun Yat-sen University), Wenqi Ren (Sun Yat-sen University), Yi Liu (Baidu Inc.), Xiaochun Cao (Sun Yat-sen University). <br>
> In  ICCV2023.


## Prerequisites

### Recommended Environment
* Python 3.7
* Pytorch 1.10
* CUDA 11.3

Other dependencies refer to requirements.txt

### Data Preparation

Datas can be download from Baidu cloud disk [[url]](https://pan.baidu.com/s/15WjlGRhYOtVNRYTj3lfE6A) (pwd: al4m) or the official website of DIV2K, Set5, Set14, B100, Urban100 and Manga109.


## Training
```
python train.py --config ./configs/spin_light_x4.yml
```

## Testing
```
sh test_script.sh
```


## Citation

If SPIN helps your research or work, please consider citing the following works:

----------
```BibTex
@inproceedings{zhang2023lightweight,
  title={Lightweight Image Super-Resolution with Superpixel Token Interaction},
  author={Zhang, Aiping, and Ren, Wenqi and Liu, Yi and Cao, Xiaochun},
  booktitle={International Conference on Computer Vision},
  year={2023}
}
```
