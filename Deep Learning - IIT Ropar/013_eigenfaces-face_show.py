from PIL import Image
from matplotlib import pyplot as plt

def face_show(subject_number) :

    path = f"D:\\data_sets\\AT&T Database of Faces\\s{subject_number}\\"

    fig, axes = plt.subplots(2, 5)

    for i, ax in enumerate(axes.flat) :

        img = Image.open(path + f"{i+1}.pgm")

        ax.imshow(img, cmap = "gray")
        ax.axis('off')

    fig.suptitle(f"Subject {subject_number}")

    plt.show()

sub_num = int(input("Enter Subject Number(1 to 40) = "))

face_show(sub_num)