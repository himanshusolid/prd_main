from django import forms
from .models import Post

class UploadCSVForm(forms.Form):
    csv_file = forms.FileField(label='Select a CSV File')

class PostAdminForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = '__all__'