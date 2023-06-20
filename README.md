## Emotion-based Song Generation

This project focuses on generating songs based on different emotions.The facial emotions are recognized using python deeplearning libraries like mediapipe and facial recognition using Open-CV.The songs are organized using a graph data structure, where each emotion is represented as a node and each node is extended into a list contain links of songs from youtube. The current implementation includes three emotions: happy, sad, and angry. However, you have the flexibility to manually add more emotions by creating new nodes.

### System Setup

To recreate the model, follow these steps:

1. Run the `data_collection.py` script and enter the name of the emotion for which you want to collect data.
2. Start the camera and display the particular emotion for 99 frames or iterations. This step allows you to gather sufficient data for training the model.
3. Repeat step 1 if you want to add more emotions to the model.
4. Once all the emotion data is collected, run the `data_training.py` script to train the model using the collected data.
5. After completing the data training step, proceed to the `main.py` file.

### Configuration in `main.py`

In the `main.py` file, perform the following steps:

1. Manually add the emotions as nodes in the graph data structure.
2. Modify the comparisons for the current emotions in the last part of the file.
3. The comparison is used to determine the major emotion based on the number of iterations (default is set to 50). This helps identify the dominant emotion during the song generation process.
4. The count of each emotion is tallied, and the majority emotion is selected. The corresponding song for that emotion is played.

### Customizing Songs

To customize the songs according to your preference, you can modify the choice of songs by changing the YouTube links provided in the emotion-specific song lists. Make sure to update the links for the particular emotion you want to customize.

### Running the Program

After completing all the necessary steps and customizations, you can run `main.py` to witness the magic of emotion-based song generation!

Feel free to explore, experiment, and add more emotions to enhance the variety of songs generated by the model.

Have fun with your emotion-based song generation project!
