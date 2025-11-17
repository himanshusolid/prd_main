(function () {
    function $(selector) {
      return document.querySelector(selector);
    }
    function $all(selector) {
      return Array.prototype.slice.call(document.querySelectorAll(selector));
    }
  
    // ALL fields you may want to toggle
    const ALL_FIELDS = [
      'master_prompt',
      'title_prompt',
      'intro_prompt',
      'modular_sections_prompt',
      'foodlove_card_prompt',
    ];
  
    // Which fields should be visible for each template
    const TEMPLATE_FIELDS = {
      regular: ['master_prompt', 'title_prompt', 'intro_prompt'],
      modular: [
        'master_prompt',
        'title_prompt',
        'intro_prompt',
        'modular_sections_prompt',
      ],
      food_love: [
        'master_prompt',
        'title_prompt',
        'intro_prompt',
        'foodlove_card_prompt',
      ],
    };
  
    function getFieldRow(fieldName) {
      // Django admin wraps each field in a div like .form-row.field-<name>
      return (
        $('.form-row.field-' + fieldName) || // older Django
        $('.field-' + fieldName)             // newer Django
      );
    }
  
    function updateVisibility() {
      const select = $('#id_template_type');
      if (!select) return;
  
      const val = select.value;
      const visible = TEMPLATE_FIELDS[val] || [];
  
      ALL_FIELDS.forEach(function (field) {
        const row = getFieldRow(field);
        if (!row) return;
  
        if (visible.indexOf(field) !== -1) {
          row.style.display = '';
        } else {
          row.style.display = 'none';
        }
      });
    }
  
    document.addEventListener('DOMContentLoaded', function () {
      const select = $('#id_template_type');
      if (!select) return;
  
      // Change handler: when user picks template type
      select.addEventListener('change', updateVisibility);
  
      // Initial run (covers both "add" and "edit" pages)
      updateVisibility();
    });
  })();
  