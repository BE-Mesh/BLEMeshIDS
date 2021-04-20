Intrusion Detection System for Bluetooth Low Energy Mesh Networks: data gathering and experimental evaluations
===================================

<img align="left" src="https://www.uniroma1.it/sites/default/files/images/logo/sapienza-big.png"/>

<br><br><br><br><br><br><br>

***Andrea Lacava, Emanuele Giacomini, Francesco D’Alterio, Francesca Cuomo***

***Sapienza University of Rome, 00184 Rome, Italy***

*Abstract* - Bluetooth Low Energy mesh networks are emerging as new standard of short burst communications. 
While security of the messages is guaranteed thought standard encryption techniques, little has been done in terms of actively protecting the overall network in case of attacks aiming to undermine its integrity.
Although many network analysis and risk mitigation techniques are currently available, they require considerable amounts of data coming from both legitimate and attack scenarios to sufficiently discriminate among them, which often turns
into the requirement of a complete description of the traffic flowing through the network.
Furthermore, there are no publicly available datasets to this extent for BLE mesh networks, due most to the novelty of the standard and to the absence of specific implementation tools.
To create a reliable mechanism of network analysis suited for BLE in this paper we propose a machine learning Intrusion Detection System (IDS) based on pattern classification and
recognition of the most classical denial of service attacks affecting this kind of networks, working on a single internal node, thus requiring a small amount of information to operate.
Moreover, in order to overcome the gap created by the absence of data, we present our data collection system based on ESP32 that allowed the collection of the packets from the Network and the Model layers of the BLE Mesh stack, together with a set of experiments conducted to get the necessary data to train the IDS.
In the last part, we describe some preliminary results obtained by the experimental setups, focusing on its strengths, as well as on the aspects where further analysis is required, hence proposing some improvements of the classification model as future work.
