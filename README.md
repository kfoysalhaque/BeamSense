# BeamSense: Rethinking Wireless Sensing with MU-MIMO Wi-Fi Beamforming Feedback

This is the implementation of the paper [BeamSense: Rethinking Wireless Sensing with MU-MIMO Wi-Fi Beamforming Feedback](https://doi.org/10.1016/j.comnet.2024.111020). The repository shares both the datasets and the source code of **BeamSense.**

If you find the project useful and you use this code, please cite our paper:

```

@article{haque2025beamsense,
  title={BeamSense: Rethinking wireless sensing with MU-MIMO Wi-Fi beamforming feedback},
  author={Haque, Khandaker Foysal and Zhang, Milin and Meneghello, Francesca and Restuccia, Francesco},
  journal={Computer Networks},
  pages={111020},
  year={2025},
  publisher={Elsevier}
}

```
and 

```

@inproceedings{haque2024bfa,
  title={BFA-Sense: Learning Beamforming Feedback Angles for Wi-Fi Sensing},
  author={Haque, Khandaker Foysal and Meneghello, Francesca and Restuccia, Francesco},
  booktitle={2024 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)},
  pages={575--580},
  year={2024},
  organization={IEEE}
}

```

## Download Dataset

(I) clone the repository with ``` git clone git@github.com:kfoysalhaque/BeamSense.git ```  <br/>
(II) ```cd BeamSense``` <br/>
(III) Then download the [BeamSense Dataset](https://ieee-dataport.org/documents/dataset-human-activity-classification-mu-mimo-bfi-and-csi#files) within the repository. <br/>
  - If the IEEE Dataport doesn't work for you, please find the Google Drive link here: https://drive.google.com/file/d/1s_Tt2ifyLYC1We7WRjRtvTMGiRQcmqyB/view?usp=drive_link <br/>
  - The full dataset is also available on Hugging Face: https://huggingface.co/datasets/foysalhaque/BeamSense

**You can also contact me (haque.k@northeastern.edu) regarding the dataset.**


(IV) Unzip the downloaded file with ``` sudo unzip Data.zip ``` <br/>

## Extract CSI from Raw pcap Files

(I) First, move into the directory _CSI_Extraction_ with ``` cd CSI_Extraction ```
(II) Execute the matlab script _Extract_CSI.m_ with  ``` matlab -nojvm -nosplash -r "Extract_CSI; exit" ```
(III) Now split the extracted CSI to samples ( with a time window of 0.1s ) by executing _CSI_to_batches.m_ script with ``` matlab -nojvm -nosplash -r "CSI_to_batches; exit" ```

**You can go with a different time window size also. But remember to keep it the same for BFI as well**


## Extract BFI from Raw pcap Files

(I) At first, split the BFIs of different stations (STAs) by executing the shell script _Feedback_split_STAs.sh_ with ``` ./Feedback_split_STAs.sh ```
(II) Now the extracted BFIs are stored within 

```
BeamSense/Data/BFI/Processed/<'Environment'>/<'STA'>/FeedBack_Pcap
```
<br/>
Now, export the Wireshark packet Dissections as CSV (needed for time windowing). You can also use Tshark with shell.
(III) Next, move into the directory _BFI_Extraction_ with ``` cd BeamSense/BFI_Extraction/ ```
(IV) Execute the matlab script _pcap_to_bfa.m_ with  ``` matlab -nojvm -nosplash -r "pcap_to_bfa; exit" ``` to extract the beamforming feedback angles (BFAs)

(V) Now split the extracted BFAs to samples ( with a time window of 0.1s -- around 10 BFI packets ) by executing _bfa_to_batches.m_ script with ``` matlab -nojvm -nosplash -r "bfa_to_batches; exit" ```


