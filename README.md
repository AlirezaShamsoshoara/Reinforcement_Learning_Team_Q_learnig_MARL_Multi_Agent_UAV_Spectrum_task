# A solution for Dynamic Spectrum Management in Mission-Critical UAV Networks using Team Q learning as a Multi-Agent Reinforcement Learning Approach

## Paper
You can find the paper related to this code [here at IEEE](https://ieeexplore.ieee.org/abstract/document/8824917) or
You can find the preprint from the [Arxiv website](https://arxiv.org/pdf/1904.07380.pdf).

In the related, we study the problem of spectrum scarcity in a network of unmanned aerial vehicles (UAVs) during mission-critical applications such as disaster monitoring and public safety missions, where the preallocated spectrum is not sufficient to offer a high data transmission rate for real-time video-streaming. In such scenarios, the UAV network can lease part of the spectrum of a terrestrial licensed network in exchange for providing relaying service. In order to optimize the performance of the UAV network and prolong its lifetime, some of the UAVs will function as a relay for the primary network while the rest of the UAVs carry out their sensing tasks. Here, we propose a team reinforcement learning algorithm performed by the UAV’s controller unit to determine the optimum allocation of sensing and relaying tasks among the UAVs as well as their relocation strategy at each time. We analyze the convergence of our algorithm and present simulation results to evaluate the system throughput in different scenarios.

* The system model of this paper is based on:
![Alt text](/image/system.JPG)

## Code
This code is run and tested on Python 2.7 on both linux and windows machine with no issues. There is a config file in this directoy which shows all the configuration parameters such as transmit power, the grid size, number of steps, number of epochs, number of runs, etc. The number of UAVs in this study is assumed to be two. For more UAVs, the code should be altered. You can simply run the main.py file to run the code. It doesn't need any input argument, all you need to configure is available in the config.py. All dependency files are available in the root directory of this repository.

```
python main.py
```

## Required Packages
* copy
* time
* match
* Numpy
* Scipy
* Random
* matplotlib.pyplot


## Results
![Alt text](/image/throughput.JPG)
![Alt text](/image/movement.JPG)
![Alt text](/image/table.JPG)

## Citation
If you find the code or the article useful, please cite our paper using this BibTeX:
```
@inproceedings{shamsoshoara2019solution,
  title={A solution for dynamic spectrum management in mission-critical UAV networks},
  author={Shamsoshoara, Alireza and Khaledi, Mehrdad and Afghah, Fatemeh and Razi, Abolfazl and Ashdown, Jonathan and Turck, Kurt},
  booktitle={2019 16th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```

## License
For academtic and non-commercial usage 
