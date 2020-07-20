import requests
from inference import generateCaption

filePath1 = 'D:\\Datasets\\cv\\ICWithAttention\\img\\test.jpg'
filePath2 = 'D:\\Datasets\\cv\\ICWithAttention\\img\\test_img.jpeg'
captionGenerator = generateCaption.CaptionGenerator()

content_type = 'image/jpeg'
files = {'file': ('dog_sleeping.jpg', open(filePath1, 'rb'), content_type)}
print(requests.post('http://localhost:5000/inference', files=files).content)

content_type = 'image/jpeg'
files = {'file': ('bus_depot.jpeg', open(filePath2, 'rb'), content_type)}
print(requests.post('http://localhost:5000/inference', files=files).content)

print('Sample output by directly invoking Caption Generator')
inference = captionGenerator.generateCaption(filePath1)
print(inference)

inference = captionGenerator.generateCaption(filePath2)
print(inference)