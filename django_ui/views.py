import base64

from io import BytesIO
import os
from django_ui.runner_wrapper import predict
from PIL import Image
from django.db.models import Model
from django.http import HttpResponse
from django.template import loader

import pdb

def index(request):
    template = loader.get_template('templates/index.html')
    context = {
        'settings': ""
    }
    return HttpResponse(template.render(context, request))


def run(request):
    image_path = request.FILES['image_path']
    nn_path = request.FILES['nn_path']

    with open('nn.hdf5', 'wb+') as destination:
        for chunk in nn_path.chunks():
            destination.write(chunk)

    with open('image_before.jpg', 'wb+') as destination:
        for chunk in image_path.chunks():
            destination.write(chunk)

    template = loader.get_template('templates/index.html')
    predict(64, 64, 15, os.getcwd() + "\\nn.hdf5", os.getcwd() + "\\image_before.jpg")

    image_after = Image.open(os.getcwd() + "\\image_after.jpg")
    response = HttpResponse(content_type='image/jpg')
    image_after.save(response, "JPEG")
    response['Content-Disposition'] = 'attachment; filename="PROCESSED_IMAGE.jpg"'
    return response
