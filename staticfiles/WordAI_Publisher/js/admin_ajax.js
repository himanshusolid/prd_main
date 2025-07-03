// static/WordAI_Publisher/js/admin_ajax.js
(function waitForDjangoJQuery() {
    if (typeof window.django !== "undefined" && typeof window.django.jQuery !== "undefined") {
        var $ = window.django.jQuery;
        $(function() {
            $('#changelist-form').on('submit', function(e) {
                var action = $('#action').val();
                console.log('Form submitted, action:', action);

                if (action === 'regenerate_featured_image' || action === 'regenerate_style_images') {
                    e.preventDefault();
                    alert('Thank you, your request is being processed.');

                    var form = $(this);
                    $.ajax({
                        url: form.attr('action'),
                        type: form.attr('method'),
                        data: form.serialize(),
                        success: function(response) {
                            console.log('AJAX success');
                        },
                        error: function(xhr, status, error) {
                            alert('There was an error processing your request.');
                        }
                    });
                }
            });
        });
    } else {
        setTimeout(waitForDjangoJQuery, 50); // Try again in 50ms
    }
})();