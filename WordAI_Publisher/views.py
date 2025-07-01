import csv
from django.shortcuts import render, redirect
from .forms import UploadCSVForm
from .models import Keyword
from django.contrib import messages

def upload_csv(request):
    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            try:
                decoded_file = csv_file.read().decode('utf-8').splitlines()
                reader = csv.DictReader(decoded_file)
                for row in reader:
                    Keyword.objects.create(
                        keyword=row.get('keyword', ''),
                        prompt_id=row.get('prompt_id', ''),
                        model_id=row.get('model_id', '')
                    )
                messages.success(request, "CSV uploaded successfully!")
            except Exception as e:
                messages.error(request, f"Error processing file: {str(e)}")
            return redirect('upload_csv')
        else:
            messages.error(request, "Invalid form submission.")
    else:
        form = UploadCSVForm()
    return render(request, 'WordAI_Publisher/upload.html', {'form': form})