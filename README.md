## About The Project

We develop a backmapping algorithm called ART-SM to convert coarse-grained molecules to atomistic resolution. Our artificial intelligence and fragment-based approach learns the Boltzmann distribution from atomistic simulations. Thus, it does not solely rely on the subsequent MD simulations, which may get trapped in local minima, to restore it in the backmapping process. Moreover, in contrast to traditional approaches ART-SM uses more than one rigid structure per fragment and selects the most appropriate one based on the coarse-grained conformation. Currently, this project is a 'proof of principle', meaning that is tested for small molecules of up to three beads and should be used with caution for larger molecules.

### Installation

1. Clone the repo
   ```sh
   git clone #TODO
   ```
2. Go into the directory
   ```sh
   cd ARTSM
   ```
3. Install
   ```sh
   pip install .
   ```
4. If the dependencies are not automatically installed
   ```sh
   pip install -r requirements.txt
   ```

## Getting Started

The best way to get started is to read and try out our tutorial. The directory `tutorial` contains the relevant MD data and a pdf with information and instructions.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Christian Pfaendner - christian.pfaendner@simtech.uni-stuttgart.de

Viktoria Korn - viktoria.korn@simtech.uni-stuttgart.de

Benjamin Unger - benjamin.unger@simtech.uni-stuttgart.de

Kristyna Pluhackova - kristyna.pluhackova@simtech.uni-stuttgart.de

