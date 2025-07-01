from django import forms
from ckeditor_uploader.widgets import CKEditorUploadingWidget
from .models import Post

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField()

class PostAdminForm(forms.ModelForm):
    generated_intro = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_style_section = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_conclusion = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    # meta_description will use the default widget (Textarea)

    class Meta:
        model = Post
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = getattr(self, 'helper', None)
        if self.instance and self.instance.pk:
            self.fields['regenerate_content'] = forms.BooleanField(
                label='Regenerate Content', required=False, initial=False,
                help_text='Check to regenerate content for this post when saving.'
            ) 