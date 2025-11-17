(function () {
    function $(selector) {
      return document.querySelector(selector);
    }
  
    const ALL_FIELDS = [
      'master_prompt', 'title_prompt', 'intro_prompt', 'style_prompt',
      'quick_style_snapshot_prompt', 'quick_style_snapshot_image_prompt',
      'daytime_outfits_prompt', 'daytime_outfits_image_prompt',
      'evening_and_nightlife_prompt', 'evening_and_nightlife_image_prompt',
      'outdoor_activities_prompt', 'outdoor_activities_image_prompt',
      'seasonal_variations_prompt', 'seasonal_variations_image_prompt',
      'packing_essentials_checklist_prompt', 'packing_essentials_checklist_image_prompt',
      'style_tips_for_blending_prompt', 'style_tips_for_blending_image_prompt',
      'destination_specific_extras_prompt', 'destination_specific_extras_image_prompt',
      'foodlove_card_prompt',
      'conclusion_prompt', 'meta_data_prompt',
      'featured_image_prompt', 'image_prompt'
    ];
  
    const TEMPLATE_FIELDS = {
      // regular : only these fields visible
      regular: [
        'master_prompt', 'title_prompt', 'intro_prompt', 'style_prompt',
        'conclusion_prompt', 'meta_data_prompt',
        'featured_image_prompt', 'image_prompt'
      ],
  
      // modular
      modular: [
        'master_prompt', 'title_prompt', 'intro_prompt',
        'quick_style_snapshot_prompt', 'quick_style_snapshot_image_prompt',
        'daytime_outfits_prompt', 'daytime_outfits_image_prompt',
        'evening_and_nightlife_prompt', 'evening_and_nightlife_image_prompt',
        'outdoor_activities_prompt', 'outdoor_activities_image_prompt',
        'seasonal_variations_prompt', 'seasonal_variations_image_prompt',
        'packing_essentials_checklist_prompt', 'packing_essentials_checklist_image_prompt',
        'style_tips_for_blending_prompt', 'style_tips_for_blending_image_prompt',
        'destination_specific_extras_prompt', 'destination_specific_extras_image_prompt',
        'conclusion_prompt', 'meta_data_prompt',
        'featured_image_prompt', 'image_prompt'
      ],
  
      // food_love
      food_love: [
        'master_prompt', 'title_prompt', 'intro_prompt', 'style_prompt',
        'foodlove_card_prompt',
        'conclusion_prompt', 'meta_data_prompt',
        'featured_image_prompt', 'image_prompt'
      ]
    };
  
    function getFieldRow(fieldName) {
      // works on most Django admin versions
      return (
        document.querySelector('.form-row.field-' + fieldName) ||
        document.querySelector('.field-' + fieldName)
      );
    }
  
    function updateVisibility() {
      const select = $('#id_template_type');
      if (!select) return;
  
      const selected = select.value;  // 'regular', 'modular', 'food_love'
      const visible = TEMPLATE_FIELDS[selected] || [];
  
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
  
      // show correct fields when page loads (edit + add)
      updateVisibility();
  
      // update when user changes the select
      select.addEventListener('change', updateVisibility);
    });
  })();
  