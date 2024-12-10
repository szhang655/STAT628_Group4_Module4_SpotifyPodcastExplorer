# Fall24_STAT628_Group4_Module4_SpotifyPodcastExplorer

## Authors 
- Ruiyuan Ming 
- Shixin Zhang

(in alphabetical order) 


## Overview
Spotify is a leading platform for music and podcast streaming. This project aims to construct new and diverse metrics tailored for Spotify's podcast episodes. 



## Repository Structure
- `data/`: Contains the raw and cleaned datasets used in the analysis.
- `code/`: Includes all codes used for data cleaning, analysis, and model building.
- `shiny/`: The code for the Shiny app, allowing real-time body fat predictions.
  
## Additional Resources

Due to the structure of this repo and the upload limit for certain files on GitHub, some other files are stored in our shared drive, inlcuding dataset we used during the process and some helper code files. Feel free to request access to our shared drives to check out.
[[https://drive.google.com/drive/folders/0AOnlxgAvrj23Uk9PVA](https://drive.google.com/drive/folders/0AOyskU3MbcYlUk9PVA)] - shared drive



## Statistical Analysis
The metrics in this analysis are derived from a Latent Dirichlet Allocation (LDA) model trained on preprocessed text data in a bag-of-words format. Each metric represents one of the eight identified topics, based on topic distribution vectors indicating the probability of a document belonging to each topic. 

## Shiny App
The Shiny app provides an interactive platform for exploring and analyzing podcast data retrieved from Spotify's API. Users can visualize key metrics and trends, compare podcasts across diverse categories such as "news," "education," and "comedy," and cluster episodes based on custom metrics. Additionally, users can explore topic modeling and clustering results, gaining insights into how podcasts align across thematic or stylistic dimensions. The app also finds the nearest podcasts with similar topics in our database. To run this shiny app, download all files in the shiny folder and type in the command line:
  ```bash
  shiny run --app-dir /path-to-your-folder
  ```

The Shiny app is also deployed to an online platform and can be accessed here:
[[https://connect.posit.cloud/szhang655/content/0193176a-818a-32e5-37d9-ca920e9e3bf6](https://0193adbc-95a6-97f5-a1d1-83338735bb2e.share.connect.posit.cloud/)]
The source code for the online version is [[https://github.com/szhang655/FlightSchedulePredictionShiny.git](https://github.com/szhang655/Podcast_Explorer_Shiny/tree/main)].


## Contact
If you have any questions or issues regarding the analysis or the app, feel free to contact us:

  Email: rming@wisc.edu, szhang655@wisc.edu

 
**Special Thanks to Professor Kang**

