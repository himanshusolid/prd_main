function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

document.addEventListener('DOMContentLoaded', function() {
    const fields = [
        {selector: '[name="generated_title"]', buttonId: 'regen-title-btn', type: 'title'},
        {selector: '[name="generated_intro"]', buttonId: 'regen-intro-btn', type: 'intro'},
        {selector: '[name="generated_style_section"]', buttonId: 'regen-style-btn', type: 'style_section'},
        {selector: '[name="generated_conclusion"]', buttonId: 'regen-conclusion-btn', type: 'conclusion'},
        {selector: '[name="meta_description"]', buttonId: 'regen-meta-description-btn', type: 'meta_description'},
        {selector: '[name="meta_title"]', buttonId: 'regen-meta-title-btn', type: 'meta_title'}
      
    ];

    fields.forEach(item => {
        const field = document.querySelector(item.selector);
        const btn = document.getElementById(item.buttonId);
        if (field && btn) {
            const row = field.closest('.form-row, .form-group');
            if (row) {
                // Create loader span to show status
                const loader = document.createElement('span');
                loader.style.marginLeft = "10px";
                loader.style.fontSize = "12px";
                loader.style.display = "none";
                loader.innerHTML = "⏳ Generating...";
                row.appendChild(loader);

                row.appendChild(btn);

                btn.addEventListener('click', function() {
                    regenerateContent(item.type, loader);
                });
            }
        }
    });

    function regenerateContent(contentType) {
        const loaderOverlay = document.getElementById('fullPageLoader');
    
        // Works for URLs like /admin/WordAI_Publisher/post/123/change/
        const match = window.location.pathname.match(/\/post\/(\d+)\/change\//);
        const postId = match ? match[1] : null;
        const csrfToken = getCookie('csrftoken');
    
        loaderOverlay.style.display = "flex";
    
        fetch(`/admin/WordAI_Publisher/keyword/ajax-regenerate-versions/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({
                post_id: postId,
                content_type: contentType
            })
        })
        .then(response => response.json())
        .then(data => {
            loaderOverlay.style.display = "none";
    
            if (data.success) {
                if (contentType === 'title') {
                    document.querySelector('[name="generated_title"]').value = data.content;
                } else if (contentType === 'intro') {
                    CKEDITOR.instances['id_generated_intro'].setData(data.content);
                } else if (contentType === 'style_section') {
                    CKEDITOR.instances['id_generated_style_section'].setData(data.content);
                } else if (contentType === 'conclusion') {
                    CKEDITOR.instances['id_generated_conclusion'].setData(data.content);
                } else if (contentType === 'meta_title') {
                    document.querySelector('[name="meta_title"]').value = data.meta_title;
                } else if (contentType === 'meta_description') {
                document.querySelector('[name="meta_description"]').value = data.meta_description;
            }
            } else {
                alert("⚠️ Error generating content.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loaderOverlay.style.display = "none";
            alert("⚠️ Failed to generate content.");
        });
    }
    

    const regenFeaturedBtn = document.getElementById('regen-featured-btn');
    if (regenFeaturedBtn) {
        regenFeaturedBtn.addEventListener('click', function() {
            regenerateFeaturedImage();
        });
    }
    
    function regenerateFeaturedImage() {
        const loaderOverlay = document.getElementById('fullPageLoader');
    
        const match = window.location.pathname.match(/\/post\/(\d+)\/change\//);
        const postId = match ? match[1] : null;
        const csrfToken = getCookie('csrftoken');
    
        loaderOverlay.style.display = "flex";
    
        // Immediately swap to placeholder image
        const img = document.querySelector('[data-featured-image-section] img');
        if (img) {
            img.src = '/static/WordAI_Publisher/img/placeholder.png';
        }
    
        fetch(`/admin/WordAI_Publisher/keyword/ajax-regenerate-versions/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({
                post_id: postId,
                content_type: 'featured_image'
            })
        })
        .then(response => response.json())
        .then(data => {
            setTimeout(() => {
                loaderOverlay.style.display = "none";
                alert("✅ Images are generating in background.\nPlease refresh the page after a few seconds.");
            }, 2000);
    
            if (!data.success) {
                alert("⚠️ Error starting featured image regeneration.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loaderOverlay.style.display = "none";
            alert("⚠️ Failed to start featured image regeneration.");
        });
    }
});
document.addEventListener('DOMContentLoaded', function() {
    const styleButtons = document.querySelectorAll('.regen-style-single-btn');
    styleButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const styleName = btn.getAttribute('data-style-name');
            regenerateSingleStyleImage(styleName, btn);
        });
    });
});

function regenerateSingleStyleImage(styleName, button) {
    const loaderOverlay = document.getElementById('fullPageLoader');

    const match = window.location.pathname.match(/\/post\/(\d+)\/change\//);
    const postId = match ? match[1] : null;
    const csrfToken = getCookie('csrftoken');

    loaderOverlay.style.display = "flex";

    // Replace only the image next to this button
    const img = button.closest('div').querySelector('img');
    if (img) {
        img.src = '/static/WordAI_Publisher/img/placeholder.png';
    }

    fetch(`/admin/WordAI_Publisher/keyword/ajax-regenerate-single-style/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({
            post_id: postId,
            style_name: styleName
        })
    })
    .then(response => response.json())
    .then(data => {
        setTimeout(() => {
            loaderOverlay.style.display = "none";
            alert(`✅ Image "${styleName}" is generating in background.\nPlease refresh the page after a few seconds.`);
        }, 10000);

        if (!data.success) {
            alert("⚠️ Error starting regeneration for " + styleName);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        loaderOverlay.style.display = "none";
        alert("⚠️ Failed to regenerate " + styleName);
    });
}