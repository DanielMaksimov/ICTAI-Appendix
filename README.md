# ICTAI-Appendix
This repository contains scripts allowing to better understand and replicate the contents of my scientific article.
This is the code I developed in order to produce its figures and results.

The data folder contains dataset files used in the scripts.
These files should be downloaded and their paths specified in the scripts.
The files consists of filtered text data and user IDs. Only users with 6 or more tweets geolocalized in the area of London are selected, and with them their tweets.

The file bert_topic_analysis.py contains the code used to produce the heatmap and the histograms.

The file london_geo_analysis.py contains the code used to produce the Voronoi regions and get the geographical clusters.

The file requirements.txt gives the details to reproduce a Python environment similar to the one I have, in order to replicate my algorithms.
Use ```pip install -r requirements.txt``` in your Python environment to install all needed libraries.
