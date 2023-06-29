# multi-emo2video

[![multi-emo2videfx](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RF1U3JN_QFikYXkzh1CtjCUuk4xHpO8d?usp=sharing)

This code utilizes a face detection and an emotion recognition model to analyze the emotions expressed in a given video ("emotion_video_path"). Based on these emotions, it applies certain effects to another provided video ("effect_video_path"). For instance, if the dominant emotion is 'happy', the script increases the brightness of the frames in the 'effect' video, if 'sad' it converts the frames to grayscale, and so forth. The processed video is then saved at the specified "output" path. The frequency at which these effects are calculated and applied depends on the provided 'interval' argument in seconds.

----------------------------------------------

The concept of the Emotion-recognition-colab tool originated at the Academy of Performing Arts in Prague as part of the project "Generative neural networks and human imagination: participatory approaches to synthetic media production" with the support of the Institutional Endowment for the Longterm Conceptual Development of the Research Institutes, as provided by the Ministry of Education, Youth and Sports of the Czech Republic in the year 2023.

Development: Scy Heidekamp - https://scyheidekamp.nl/ Concept & research: Lenka Hamosova - https://hamosova.com/
