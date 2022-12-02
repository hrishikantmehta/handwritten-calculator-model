import streamlit as st
from streamlit_drawable_canvas import st_canvas

import numpy as np
import cv2
import keras

model = keras.models.load_model("model.h5")

st.title('Handwritten Calculator')

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=5,
    stroke_color="rgba(0,0,0,1)",
    background_color="rgba(255,255,255,1)",
    background_image=None,
    update_streamlit=True,# for realtime update
    height=150,
    width=1000,
    drawing_mode="freedraw",
    key="canvas",
)

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

button=st.button('solve')

if button:

    with st.spinner('wait'):
        img=canvas_result.image_data

        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img=~img
        _,thrsh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        cntr,_=cv2.findContours(thrsh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        digits=[]

        for i in range(len(cntr)):
            rect=cv2.boundingRect(cntr[i])

            digits.append([rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]])

        digits=sorted(digits)

        # handle the division operator
        new_digits=[]

        for i in range(len(digits)):
            if i==0 or new_digits[-1][2]<digits[i][0]:
                new_digits.append(digits[i])
            else:
                new_digits[-1][0]=min(new_digits[-1][0],digits[i][0])
                new_digits[-1][1]=min(new_digits[-1][1],digits[i][1])
                new_digits[-1][2]=max(new_digits[-1][2],digits[i][2])
                new_digits[-1][3]=max(new_digits[-1][3],digits[i][3])

        expression=''

        for i in range(len(new_digits)):
            img_crop=img[new_digits[i][1]:new_digits[i][3],new_digits[i][0]:new_digits[i][2]]

            # plt.imshow(img_crop)

            # width>height
            if(img_crop.shape[1]>img_crop.shape[0]):
                scale=int(40*100/img_crop.shape[1])
            else:
                scale=int(40*100/img_crop.shape[0])

            width=int(img_crop.shape[1]*scale/100)
            height=int(img_crop.shape[0]*scale/100)

            img_resize=cv2.resize(img_crop,(width,height))

            # padding
            m=img_resize.shape[0]
            n=img_resize.shape[1]

            left=(40-n)//2
            right=40-n-left
            up=(40-m)//2
            down=40-m-up

            # padding up
            for _ in range(up):
                img_resize=np.insert(img_resize,0,np.array([0 for i in range(n)]),axis=0)

            # padding down
            for _ in range(down):
                img_resize=np.insert(img_resize,img_resize.shape[0],np.array([0 for i in range(n)]),axis=0)

            # padding left
            for _ in range(left):
                img_resize=np.insert(img_resize,0,np.array([0 for i in range(40)]),axis=1)

            # padding right
            for _ in range(right):
                img_resize=np.insert(img_resize,img_resize.shape[1],np.array([0 for i in range(40)]),axis=1)

            # img_resize=img_resize/255.0

            pred=model.predict([img_resize.reshape(-1, 40, 40, 1)])
            pred=np.argmax(pred, axis = 1)

            if pred[0]<10:
                expression+=str(pred[0])
            else:
                if pred[0]==10:
                    expression+=str(' + ')
                elif pred[0]==11:
                    expression+=str(' - ')
                elif pred[0]==12:
                    expression+=str(' * ')
                else:
                    expression+=str(' / ')

        # print('Predicted expression is : '+expression)

        try:
            st.write(expression+' = '+str(round(eval(expression),2)))
        except:
            st.error('Invalid Expression')
        # print(expression+' = '+str(eval(expression)))