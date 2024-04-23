<!-- PROJECT SHIELDS -->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# ART-SM: Boosting Fragment-based Backmapping by Machine Learning

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contacts">Contacts</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project                                                                                    

We develop a backmapping algorithm called ART-SM to convert coarse-grained molecules to atomistic resolution. Our artificial intelligence and fragment-based approach learns the Boltzmann distribution from atomistic simulations. Thus, it does not solely rely on MD simulations, which may get trapped in local minima, to restore the Boltzmann distribution in the backmapping process. Moreover, in contrast to traditional approaches ART-SM uses more than one rigid structure per fragment and selects the most appropriate one based on the coarse-grained conformation. ART-SM has two main steps:

1. Build a fragment pair database from atomistic simulation data
2. Backmap a CG structure to atomistic resolution using the database built in step 1

Check out our tutorial in the corresponding `tutorial` folder to try it out!  
Note that this project is a 'proof of principle', meaning that is tested for small molecules of up to three beads and should be used with caution for larger or complex molecules. It will be extended to lipids, macromolecules and ring structures in the future.

<!-- INSTALLATION -->
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/chrispfae/ART-SM.git
   ``` 
2. Go into the directory
   ```sh
   cd ART-SM
   ``` 
3. Install
   ```sh
   pip install .
   ``` 
4. If the dependencies are not automatically installed
   ```sh
   pip install -r requirements.txt
   pip install .
   ``` 
Note that at least Python version 3.9 is required.

<!-- GETTING STARTED -->
## Getting Started

The best way to get started is to read and try out the tutorial. The directory `tutorial` contains the relevant MD data and a pdf with information and instructions.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACTS -->
## Contacts

Christian Pfaendner - christian.pfaendner@simtech.uni-stuttgart.de

Viktoria Korn - viktoria.korn@simtech.uni-stuttgart.de

Benjamin Unger - benjamin.unger@simtech.uni-stuttgart.de

Kristyna Pluhackova - kristyna.pluhackova@simtech.uni-stuttgart.de

Project Link: [https://github.com/chrispfae/ART-SM.git](https://github.com/chrispfae/ART-SM.git)

[license-shield]: https://img.shields.io/github/license/chrispfae/ART-SM.svg?style=for-the-badge
[license-url]: https://github.com/Jonas-Nicodemus/PINNs-based-MPC/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/christian-pfaendner-ba1a53226/
