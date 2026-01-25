
import streamlit as st 

def analyze_detections(results, conf):
    """Analyze detection results and categorize them"""
    boxes = results[0].boxes

    stats = {
        'total_detections': 0,
        'with_mask': 0,
        'without_mask': 0,
        'detections': []
    }

    # Process all boxes (YOLO already filtered by confidence)
    for box in boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        cls_name = results[0].names[cls]

        # Add to detections list
        stats['detections'].append({
            'class': cls_name,
            'confidence': conf,
            'bbox': box.xyxy[0].tolist()
        })

        stats['total_detections'] += 1

        # Categorize based on class name - adjust these patterns to match YOUR model's class names
        cls_lower = cls_name.lower()

        # Check for "with mask" variations
        if any(keyword in cls_lower for keyword in ['with_mask', 'with mask', 'mask_on', 'wearing_mask', 'masked']):
            stats['with_mask'] += 1
        # Check for "without mask" / "no mask" variations
        elif any(keyword in cls_lower for keyword in ['without_mask', 'without mask', 'no_mask', 'no mask', 'nomask', 'mask_off']):
            stats['without_mask'] += 1
        # If class name just says "mask", assume it means "with mask"
        elif cls_lower == 'mask':
            stats['with_mask'] += 1
        # Default fallback - you may need to adjust this
        else:
            # Print to help debug what class names your model uses
            print(f"⚠️ Unknown class detected: '{cls_name}' - defaulting to 'without_mask'")
            stats['without_mask'] += 1

    return stats


def render_analytics(stats, compliance):
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{stats["total_detections"]}</div><div class="metric-label">Total</div></div>',
                    unsafe_allow_html=True,
                )
    with metric_cols[1]:
        st.markdown(
                    f'<div class="metric-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);"><div class="metric-value">{stats["with_mask"]}</div><div class="metric-label">With Mask</div></div>',
                    unsafe_allow_html=True,
                )
    with metric_cols[2]:
        st.markdown(
                    f'<div class="metric-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);"><div class="metric-value">{stats["without_mask"]}</div><div class="metric-label">Without Mask</div></div>',
                    unsafe_allow_html=True,
                )
    with metric_cols[3]:
        st.markdown(
                    f'<div class="metric-card" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);"><div class="metric-value">{compliance:.1f}%</div><div class="metric-label">Compliance</div></div>',
                    unsafe_allow_html=True,
                )


