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
    # 🆕 Add all 8 new modular CKEditor fields
    generated_quick_style_snapshot = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_packing_essentials_checklist = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_daytime_outfits = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_evening_and_nightlife = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_outdoor_activities = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_seasonal_variations = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_style_tips_for_blending = forms.CharField(widget=CKEditorUploadingWidget(), required=False)
    generated_destination_specific_extras = forms.CharField(widget=CKEditorUploadingWidget(), required=False)

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