from darkflow.net.build import TFNet
import numpy as np
import math
import cv2
import os
import json
import base64
import requests
import skvideo.io
import PIL
import sys


def dectect_cars_per_image(cfg, weights, im):
    #demo purpose
    init_url='https://api.belairdirect.com/quickquote-blr/initDataContext?company=BELAIR&province=QC&language=EN&distributor=BEL&platform=desktop&partnershipId='
    headers = {}
    headers['Origin'] = 'https://apps.belairdirect.com'
    headers['Referer'] = 'https://apps.belairdirect.com/quick-quote/desktop/index.html?prov=qc&lang=en&intcid=homepage'
    params = {}
    params['distributor'] = 'BEL'
    params['company'] = 'BELAIR'
    params['province'] = 'QC'
    params['language'] = 'EN'
    params['emailAddress'] = ''
    params['licenceNumber'] = ''
    params['firstNameLicence'] = ''
    params['year'] = '2012'
    #params['make'] = 'HONDA'
    #params['model'] = '021001'
    params['distanceWorkSchool'] = '10'
    params['annualKm'] = '14000'
    params['gender'] = 'M'
    params['firstName'] = 'test'
    params['lastName'] = 'test'
    params['dateOfbirthYear'] = '1996'
    params['dateOfbirthMonth'] = '12'
    params['dateOfbirthDay'] = '12'
    params['homePhoneNumber'] = '514-555-1212'
    params['postalCode'] = 'H2W 1X9'
    params['firstLicencedAt'] = '18'
    params['yearsWithCurrentInsurer'] = '1'
    params['marketingConsent'] = 'false'
    params['creditScore'] = '0'
    params['otherAntiTheftDeviceIndicator'] = 'false'
    valid_makes = ['VOLKSWAGEN','FORD','DODGE','HONDA',
        'TOYOTA','TOYOTA','MAZDA','BMW','NISSAN','HYUNDAI']
    valid_models = ['Golf', 'F150', 'Grand Caravan','CR-V',
        'RAV-4','Camry','MAZDA3','328i','Altima','Elantra']
    valid_code =['968900','355801','266200','027101','755700',
        '045002','758600','903501','091005','052806']
    qq_price = {}
    #since we only use once
    mock_premium = ['2611.00',
                    '3171.00',
                    '2286.00',
                    '2825.00',
                    '2872.00',
                    '3036.00',
                    '2420.00',
                    '3701.00',
                    '2657.00',
                    '3204.00']
    index = 0
    for model in valid_models:
        #resp = requests.get(url=init_url,headers=headers)
        #policyVersionId =json.loads(resp.content.decode("utf-8"))['body']['policyVersionId']
        #params['policyVersionId'] = policyVersionId
        params['make'] = valid_makes[index]
        params['model'] = valid_code[index]
        #response = requests.get(url='https://api.belairdirect.com/quickquote-blr/getPrice',
        #    params=params, headers=headers)
        #premium = json.loads(response.content.decode("utf-8"))['body']['vehicles'][0]['offers']['CUSTOM']['offerDetails']['priceYearly']
        #print(valid_makes[index] +valid_code[index] + ':' + premium)
        #qq_price[valid_models[index]] = premium


        qq_price[valid_models[index]] = mock_premium[index]
        index += 1

    options = {"model": cfg, "load": weights, "threshold": 0.6}


    
    #loading yolo model
    tfnet = TFNet(options)
    
    imgcv = im
    
    boxes = tfnet.return_predict(imgcv)
    print(boxes)
    
    h, w, _ = imgcv.shape
    car_list = []
    car_num = 1
    for b in boxes:
        left = b['topleft']['x']
        right = b['bottomright']['x']
        top = b['topleft']['y']
        bot = b['bottomright']['y']
        mess = b['label']
        confidence = b['confidence']
        thick = int((h + w) // 300)

        #output the image box
        response = 'Cannot recognize a car.'
        print_text = 'null'
        if mess == 'car':
            crop = imgcv[top:bot,left:right]
            car_path = 'test/out/tmp/tmp.jpg'
            cv2.imwrite(car_path,crop)
            car_num += 1
            #post, getting result from car model
            with open(car_path, "rb") as image_file:

                encoded_string = base64.b64encode(image_file.read())
                url ='http://52.168.131.37:8080/predict/car_model2.0'
                data= encoded_string
                headers = {}
                headers['Content-Type'] = 'application/json'
                resp = requests.post(url=url, data=json.dumps(data.decode("utf-8")),headers=headers)
                response = json.loads(resp.content.decode("utf-8"))['top5_results']['top1']
                #response = json.loads(resp.content.decode("utf-8"))['Make'] + ' ' + json.loads(resp.content.decode("utf-8"))['Model'] + ':'
                #year_list = json.loads(resp.content.decode("utf-8"))['Year']
                #response += year_list[-1]
            os.remove(car_path)
            #car_list.append(json.loads(resp.content.decode("utf-8"))['Make'] + ' ' + json.loads(resp.content.decode("utf-8"))['Model'])
            text = response['make'] + ' ' +response['model'] + ':' + response['prob']
            car_list.append(text)
            #demo purpose
            prob_of_car = float(response['prob'])
            #draw rec and text
            if response['model'] not in valid_models:
                print_text = response['make'] + ' ' +response['model'] 
            else:
            #    cv2.rectangle(imgcv,
            #        (left, top), (right, bot),
            #        (128, 127, 77), thick//2)
            #    cv2.putText(imgcv, 'car', (left, top - 12),
            #        0, 1e-3 * h * 2 / 3, (128, 127, 77),thick//5)
            #    continue
                print_text = response['make'] + ' ' +response['model'] + ': $' + qq_price[response['model']]
            if prob_of_car > 0.95:
                cv2.rectangle(imgcv,
                    (left, top), (right, bot),
                    (56, 254, 0), thick//2)
                cv2.putText(imgcv, print_text, (left, top - 12),
                    0, 1e-3 * h * 2 / 3, (56, 254, 0),thick//5)
            elif prob_of_car < 0.7:
                cv2.rectangle(imgcv,
                    (left, top), (right, bot),
                    (128, 127, 77), thick//2)
                cv2.putText(imgcv, 'car', (left, top - 12),
                    0, 1e-3 * h * 2 / 3, (128, 127, 77),thick//5)

            else:
                cv2.rectangle(imgcv,
                    (left, top), (right, bot),
                    (255, 255, 153), thick//2)
                cv2.putText(imgcv, print_text,
                    (left, top - 12),0, 1e-3 * h * 2 / 3, (255, 255, 153),thick//5)
        else:
            print("only support cars")
            return

    video_output = 'test/out/video.txt'
    with open(video_output, 'a') as f:
        f.write(json.dumps(car_list) + '\t\n')

    #return image if in video
    return imgcv

    

    
    
def image_set(cfg, weights, dir):

    outfolder = 'test/out/'
    tmpfolder = 'test/out/tmp/'
    file_list = os.listdir(outfolder)
    tmp_list = os.listdir(tmpfolder)
    for f in file_list:
        if f.endswith('.jpg'):
            os.remove(outfolder  + f)
    for f in tmp_list:
        if f.endswith('tmp.jpg'):
            os.remove(tmpfolder  + f)

    image_list = os.listdir(dir)
    print(image_list)
    for im in image_list:
        image = cv2.imread(dir + im)
        print(type(image))
        if isinstance(image, np.ndarray):
            processed = dectect_cars_per_image(cfg, weights, image)
            
            img_name = 'test/out/' + os.path.splitext(os.path.basename(im))[0] + '.jpg'
            print(img_name)
            cv2.imwrite(img_name, processed)
                                                        
outfolder = 'test/out'
tmpfolder = 'test/out/tmp'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
if not os.path.exists(tmpfolder):
    os.makedirs(tmpfolder)
image_set(sys.argv[1], sys.argv[2], 'test/')
