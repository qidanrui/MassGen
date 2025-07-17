from datasets import load_dataset
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io
import re
import random

def convert_to_json_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return str(obj)  # Convert other non-serializable objects to string

def collect_all_field_names(dataset):
    """Collect all unique field names from the entire dataset"""
    all_field_names = set()
    
    print(f"Scanning {len(dataset)} items in the dataset...")
    
    for i, item in enumerate(dataset):
        # Add all keys from current item
        all_field_names.update(item.keys())
        
        # Print progress every 1000 items
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} items...")
    
    return sorted(list(all_field_names))

def analyze_image_statistics(dataset):
    """Analyze image statistics in the dataset"""
    items_with_image = 0
    items_without_image = 0
    items_with_empty_image = 0
    
    print(f"Analyzing image statistics for {len(dataset)} items...")
    
    for i, item in enumerate(dataset):
        # Check if image field exists and has content
        if 'image' in item:
            if item['image'] and item['image'].strip():
                items_with_image += 1
            else:
                items_with_empty_image += 1
        else:
            items_without_image += 1
        
        # Print progress every 1000 items
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} items...")
    
    return {
        'with_image': items_with_image,
        'without_image': items_without_image,
        'empty_image': items_with_empty_image,
        'total': len(dataset)
    }

def display_dataset_item(item, index):
    """Display a dataset item with image if available"""
    print(f"\n{'='*60}")
    print(f"DATASET ITEM {index}")
    print(f"{'='*60}")
    
    # Display basic information
    print(f"ID: {item.get('id', 'N/A')}")
    print(f"Subject: {item.get('raw_subject', 'N/A')}")
    print(f"Category: {item.get('category', 'N/A')}")
    print(f"Author: {item.get('author_name', 'N/A')}")
    
    # Display question
    print(f"\nQUESTION:")
    print(f"{item.get('question', 'N/A')}")
    
    # Display image if available
    if 'image' in item and item['image']:
        try:
            # The image is in base64 format with data:image/jpeg;base64, prefix
            image_data = item['image']
            if image_data.startswith('data:image'):
                # Extract base64 data after the comma
                base64_data = image_data.split(',')[1]
                
                # Decode base64 to image
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Display the image
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                plt.axis('off')
                plt.title(f"Image for Question {index}")
                plt.show()
                
                print(f"Image displayed above (Size: {image.size})")
                
        except Exception as e:
            print(f"Error displaying image: {e}")
    else:
        print("No image available for this question")
    
    # Display answer and other info
    print(f"\nCORRECT ANSWER: {item.get('answer', 'N/A')}")
    print(f"ANSWER TYPE: {item.get('answer_type', 'N/A')}")
    
    if 'rationale' in item and item['rationale']:
        print(f"\nRATIONALE:")
        print(f"{item['rationale']}")

def check_for_test_case_question(dataset, test_case_file='test_case.txt'):
    """Check if any item in the dataset matches the question from test_case.txt"""
    print(f"\n{'='*60}")
    print("CHECKING FOR TEST CASE QUESTION MATCH")
    print(f"{'='*60}")
    
    try:
        # Read the test case question
        with open(test_case_file, 'r', encoding='utf-8') as f:
            test_case_question = f.read().strip()
        
        print(f"Test case question loaded from {test_case_file}")
        print(f"Question length: {len(test_case_question)} characters")
        
        # Clean the test case question for better matching
        def clean_text(text):
            """Clean text by removing extra whitespace and normalizing"""
            return re.sub(r'\s+', ' ', text.strip())
        
        cleaned_test_case = clean_text(test_case_question)
        
        # Search through all items in the dataset
        matches_found = []
        
        print(f"\nSearching through {len(dataset)} items...")
        
        for i, item in enumerate(dataset):
            if 'question' in item and item['question']:
                cleaned_dataset_question = clean_text(item['question'])
                
                # Check for exact match
                if cleaned_dataset_question == cleaned_test_case:
                    matches_found.append({
                        'index': i,
                        'item': item,
                        'match_type': 'exact'
                    })
                
                # Check for partial match (test case is substring of dataset question)
                elif cleaned_test_case in cleaned_dataset_question:
                    matches_found.append({
                        'index': i,
                        'item': item,
                        'match_type': 'partial_test_in_dataset'
                    })
                
                # Check for partial match (dataset question is substring of test case)
                elif cleaned_dataset_question in cleaned_test_case:
                    matches_found.append({
                        'index': i,
                        'item': item,
                        'match_type': 'partial_dataset_in_test'
                    })
            
            # Print progress every 1000 items
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(dataset)} items...")
        
        # Report results
        print(f"\n{'='*60}")
        print("SEARCH RESULTS")
        print(f"{'='*60}")
        
        if matches_found:
            print(f"Found {len(matches_found)} matching item(s):")
            
            for match in matches_found:
                print(f"\n{'-'*40}")
                print(f"Match Type: {match['match_type']}")
                print(f"Dataset Index: {match['index']}")
                print(f"Item ID: {match['item'].get('id', 'N/A')}")
                print(f"Subject: {match['item'].get('raw_subject', 'N/A')}")
                print(f"Category: {match['item'].get('category', 'N/A')}")
                print(f"Answer: {match['item'].get('answer', 'N/A')}")
                
                if match['match_type'] == 'exact':
                    print("✓ EXACT MATCH FOUND!")
                elif match['match_type'] == 'partial_test_in_dataset':
                    print("⚠ Test case question is contained within dataset question")
                elif match['match_type'] == 'partial_dataset_in_test':
                    print("⚠ Dataset question is contained within test case question")
                
                print(f"\nDataset Question (first 200 chars):")
                print(f"{match['item']['question'][:200]}...")
        else:
            print("❌ No matching questions found in the dataset")
            
            # Show the first few characters of the test case for reference
            print(f"\nTest case question (first 200 chars):")
            print(f"{test_case_question[:200]}...")
            
            # Suggest checking a few sample questions for comparison
            print(f"\nSample questions from dataset for comparison:")
            for i in range(min(3, len(dataset))):
                if 'question' in dataset[i] and dataset[i]['question']:
                    print(f"\nSample {i+1} (first 200 chars):")
                    print(f"{dataset[i]['question'][:200]}...")
        
        # Save results to files
        # Save text summary
        with open('test_case_search_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"Test Case Question Search Results\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Dataset: cais/hle (test split)\n")
            f.write(f"Total items searched: {len(dataset)}\n")
            f.write(f"Matches found: {len(matches_found)}\n\n")
            
            if matches_found:
                f.write("MATCHES:\n")
                for match in matches_found:
                    f.write(f"\n{'-'*30}\n")
                    f.write(f"Match Type: {match['match_type']}\n")
                    f.write(f"Dataset Index: {match['index']}\n")
                    f.write(f"Item ID: {match['item'].get('id', 'N/A')}\n")
                    f.write(f"Subject: {match['item'].get('raw_subject', 'N/A')}\n")
                    f.write(f"Category: {match['item'].get('category', 'N/A')}\n")
                    f.write(f"Answer: {match['item'].get('answer', 'N/A')}\n")
                    f.write(f"Question: {match['item']['question']}\n")
            else:
                f.write("No matches found.\n")
                f.write(f"\nTest case question:\n{test_case_question}\n")
        
        # Save original dataset items to JSON file
        if matches_found:
            original_items = []
            for match in matches_found:
                # Convert the original item to JSON serializable format
                original_item = convert_to_json_serializable(match['item'])
                
                # Add metadata about the match
                item_with_metadata = {
                    'dataset_index': match['index'],
                    'match_type': match['match_type'],
                    'original_item': original_item
                }
                original_items.append(item_with_metadata)
            
            # Save to JSON file
            with open('test_case_matched_items.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'dataset': 'cais/hle (test split)',
                        'total_items_searched': len(dataset),
                        'matches_found': len(matches_found),
                        'test_case_file': test_case_file,
                        'search_timestamp': str(np.datetime64('now'))
                    },
                    'matched_items': original_items
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\nSearch results saved to 'test_case_search_results.txt'")
            print(f"Original matched items saved to 'test_case_matched_items.json'")
        else:
            print(f"\nSearch results saved to 'test_case_search_results.txt'")
            print("No matches found - no JSON file created")
        
        return matches_found
        
    except FileNotFoundError:
        print(f"❌ Error: {test_case_file} not found")
        return []
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return []

def analyze_field_values(dataset, fields_to_analyze=['answer_type', 'category', 'raw_subject']):
    """Analyze possible values for specified fields in the dataset"""
    print(f"\n{'='*60}")
    print("ANALYZING FIELD VALUES")
    print(f"{'='*60}")
    
    field_stats = {}
    
    # Initialize dictionaries for each field
    for field in fields_to_analyze:
        field_stats[field] = {}
    
    print(f"Analyzing {len(fields_to_analyze)} fields across {len(dataset)} items...")
    
    # Process each item in the dataset
    for i, item in enumerate(dataset):
        for field in fields_to_analyze:
            if field in item:
                value = item[field]
                
                # Convert to string for consistent handling
                if value is None:
                    value_str = "None"
                elif isinstance(value, bool):
                    value_str = str(value)
                elif isinstance(value, (int, float)):
                    value_str = str(value)
                else:
                    value_str = str(value).strip()
                
                # Count occurrences
                if value_str in field_stats[field]:
                    field_stats[field][value_str] += 1
                else:
                    field_stats[field][value_str] = 1
            else:
                # Field is missing from this item
                if "MISSING_FIELD" in field_stats[field]:
                    field_stats[field]["MISSING_FIELD"] += 1
                else:
                    field_stats[field]["MISSING_FIELD"] = 1
        
        # Print progress every 1000 items
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} items...")
    
    # Display results
    print(f"\n{'='*60}")
    print("FIELD VALUE ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    for field in fields_to_analyze:
        print(f"\n{'-'*50}")
        print(f"FIELD: {field}")
        print(f"{'-'*50}")
        
        if field_stats[field]:
            # Sort by count (descending) then by value name
            sorted_values = sorted(field_stats[field].items(), 
                                 key=lambda x: (-x[1], x[0]))
            
            total_items = len(dataset)
            print(f"Total unique values: {len(field_stats[field])}")
            print(f"Value distribution:")
            
            for value, count in sorted_values:
                percentage = (count / total_items) * 100
                print(f"  {value:<30} : {count:>6,} ({percentage:>5.1f}%)")
        else:
            print(f"No values found for field '{field}'")
    
    # Save detailed results to file
    with open('field_values_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(f"Field Values Analysis - HLE Dataset\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Dataset: cais/hle (test split)\n")
        f.write(f"Total items analyzed: {len(dataset):,}\n")
        f.write(f"Fields analyzed: {', '.join(fields_to_analyze)}\n\n")
        
        for field in fields_to_analyze:
            f.write(f"\n{'-'*50}\n")
            f.write(f"FIELD: {field}\n")
            f.write(f"{'-'*50}\n")
            
            if field_stats[field]:
                sorted_values = sorted(field_stats[field].items(), 
                                     key=lambda x: (-x[1], x[0]))
                
                total_items = len(dataset)
                f.write(f"Total unique values: {len(field_stats[field])}\n")
                f.write(f"Value distribution:\n\n")
                
                for value, count in sorted_values:
                    percentage = (count / total_items) * 100
                    f.write(f"  {value:<30} : {count:>6,} ({percentage:>5.1f}%)\n")
            else:
                f.write(f"No values found for field '{field}'\n")
    
    print(f"\nDetailed field analysis saved to 'field_values_analysis.txt'")
    
    # Create a summary dictionary for potential further use
    summary = {}
    for field in fields_to_analyze:
        summary[field] = {
            'unique_values': len(field_stats[field]),
            'most_common': max(field_stats[field].items(), key=lambda x: x[1]) if field_stats[field] else None,
            'all_values': list(field_stats[field].keys()) if field_stats[field] else []
        }
    
    return field_stats, summary

def create_category_sample_subset(dataset, category_field='raw_subject', min_examples=10, samples_per_category=50):
    """Create a subset by randomly selecting multiple examples from each category with >= min_examples"""
    print(f"\n{'='*60}")
    print("CREATING CATEGORY SAMPLE SUBSET")
    print(f"{'='*60}")
    
    # Group items by category
    category_items = {}
    
    print(f"Grouping {len(dataset)} items by {category_field}...")
    
    for i, item in enumerate(dataset):
        if category_field in item and item[category_field]:
            category = str(item[category_field]).strip()
            
            if category not in category_items:
                category_items[category] = []
            
            category_items[category].append({
                'dataset_index': i,
                'item': item
            })
        
        # Print progress every 1000 items
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} items...")
    
    print(f"\nFound {len(category_items)} unique categories")
    
    # Filter categories with >= min_examples
    qualifying_categories = {k: v for k, v in category_items.items() if len(v) >= min_examples}
    
    print(f"Categories with >= {min_examples} examples: {len(qualifying_categories)}")
    
    if not qualifying_categories:
        print(f"❌ No categories found with >= {min_examples} examples")
        return []
    
    # Display category statistics
    print(f"\nQualifying categories:")
    sorted_categories = sorted(qualifying_categories.items(), key=lambda x: (x[0]))  # Sort by category name
    for category, items in sorted_categories:
        print(f"  {category:<40} : {len(items):>4} examples")
    
    # Randomly select multiple examples from each qualifying category (without images)
    selected_samples = []
    
    print(f"\nRandomly selecting up to {samples_per_category} examples from each qualifying category (without images)...")
    
    for category, items in sorted_categories:  # Use sorted categories to maintain order
        # Filter items without images
        items_without_images = []
        for item_data in items:
            item = item_data['item']
            has_image = ('image' in item and item['image'] and str(item['image']).strip())
            if not has_image:
                items_without_images.append(item_data)
        
        if items_without_images:
            # Randomly select up to samples_per_category items from those without images
            num_to_select = min(samples_per_category, len(items_without_images))
            selected_items_data = random.sample(items_without_images, num_to_select)
            
            print(f"  {category:<40} : Selected {num_to_select} indices (no images, {len(items_without_images)}/{len(items)} available)")
            
            # Add each selected item to the samples list
            for idx, selected_item_data in enumerate(selected_items_data):
                selected_item = selected_item_data['item']
                
                # Create a clean copy for JSON serialization
                sample = {
                    'dataset_index': selected_item_data['dataset_index'],
                    'category': category,
                    'category_sample_index': idx + 1,  # 1-based index within category
                    'selected_from_total': len(items),
                    'available_without_images': len(items_without_images),
                    'item_data': convert_to_json_serializable(selected_item)
                }
                
                selected_samples.append(sample)
        else:
            print(f"  {category:<40} : ⚠️  SKIPPED - no items without images ({len(items)} total)")
            continue
    
    # Save to JSON file
    output_filename = f'category_sample_subset_{category_field}_{samples_per_category}each.json'
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'dataset': 'cais/hle (test split)',
                'total_dataset_items': len(dataset),
                'category_field': category_field,
                'min_examples_threshold': min_examples,
                'samples_per_category': samples_per_category,
                'total_categories': len(category_items),
                'qualifying_categories': len(qualifying_categories),
                'total_selected_samples': len(selected_samples),
                'creation_timestamp': str(np.datetime64('now'))
            },
            'samples': selected_samples
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Successfully created subset with {len(selected_samples)} samples")
    print(f"📁 Saved to: {output_filename}")
    
    # Display summary statistics
    print(f"\nSUBSET SUMMARY:")
    print(f"{'='*40}")
    print(f"Total samples selected: {len(selected_samples)}")
    print(f"Categories included: {len(qualifying_categories)} (out of {len(category_items)} total)")
    print(f"Samples per category: up to {samples_per_category}")
    print(f"Minimum examples per category: {min_examples}")
    
    # Show category-wise breakdown
    print(f"\nCategory breakdown:")
    current_category = None
    category_count = 0
    for sample in selected_samples:
        if sample['category'] != current_category:
            if current_category is not None:
                print(f"  {current_category:<40} : {category_count} samples")
            current_category = sample['category']
            category_count = 1
        else:
            category_count += 1
    # Print the last category
    if current_category is not None:
        print(f"  {current_category:<40} : {category_count} samples")
    
    # Show some examples of what was selected
    print(f"\nFirst 5 selected samples:")
    for i, sample in enumerate(selected_samples[:5]):
        item = sample['item_data']
        print(f"\n{i+1}. Category: {sample['category']} (sample {sample['category_sample_index']})")
        print(f"   Dataset Index: {sample['dataset_index']}")
        print(f"   Question: {item.get('question', 'N/A')[:100]}...")
        print(f"   Answer: {item.get('answer', 'N/A')}")
    
    return selected_samples

dataset = load_dataset("cais/hle", split="test")

# Collect all field names from the entire dataset
print("Collecting all field names from the dataset...")
all_fields = collect_all_field_names(dataset)

print(f"\n{'='*60}")
print(f"ALL FIELD NAMES IN THE DATASET ({len(all_fields)} total)")
print(f"{'='*60}")
for i, field in enumerate(all_fields, 1):
    print(f"{i:2d}. {field}")

# Analyze image statistics
print("\n" + "="*60)
print("ANALYZING IMAGE STATISTICS")
print("="*60)
image_stats = analyze_image_statistics(dataset)

print(f"\nIMAGE STATISTICS:")
print(f"{'='*40}")
print(f"Total items in dataset: {image_stats['total']:,}")
print(f"Items with images: {image_stats['with_image']:,} ({image_stats['with_image']/image_stats['total']*100:.1f}%)")
print(f"Items with empty images: {image_stats['empty_image']:,} ({image_stats['empty_image']/image_stats['total']*100:.1f}%)")
print(f"Items without image field: {image_stats['without_image']:,} ({image_stats['without_image']/image_stats['total']*100:.1f}%)")

# Save field names to a file for reference
with open('dataset_field_names.txt', 'w') as f:
    f.write(f"Dataset: cais/hle (test split)\n")
    f.write(f"Total items: {len(dataset)}\n")
    f.write(f"Total unique field names: {len(all_fields)}\n\n")
    f.write("Field names:\n")
    for i, field in enumerate(all_fields, 1):
        f.write(f"{i:2d}. {field}\n")

print(f"\nField names saved to 'dataset_field_names.txt'")

# Save image statistics to a file
with open('dataset_image_statistics.txt', 'w') as f:
    f.write(f"Dataset: cais/hle (test split)\n")
    f.write(f"Image Statistics Analysis\n")
    f.write(f"{'='*40}\n\n")
    f.write(f"Total items in dataset: {image_stats['total']:,}\n")
    f.write(f"Items with images: {image_stats['with_image']:,} ({image_stats['with_image']/image_stats['total']*100:.1f}%)\n")
    f.write(f"Items with empty images: {image_stats['empty_image']:,} ({image_stats['empty_image']/image_stats['total']*100:.1f}%)\n")
    f.write(f"Items without image field: {image_stats['without_image']:,} ({image_stats['without_image']/image_stats['total']*100:.1f}%)\n")

print(f"Image statistics saved to 'dataset_image_statistics.txt'")

# Analyze field values
print("\n" + "="*60)
print("ANALYZING FIELD VALUES")
print("="*60)
field_stats, field_summary = analyze_field_values(dataset)

# Display summary information
print(f"\n{'='*60}")
print("FIELD ANALYSIS SUMMARY")
print(f"{'='*60}")
for field, info in field_summary.items():
    print(f"\n{field}:")
    print(f"  Unique values: {info['unique_values']}")
    if info['most_common']:
        print(f"  Most common: '{info['most_common'][0]}' ({info['most_common'][1]:,} occurrences)")
    print(f"  All values: {info['all_values'][:5]}{'...' if len(info['all_values']) > 5 else ''}")

# Create category sample subset
print("\n" + "="*60)
print("CREATING CATEGORY SAMPLE SUBSET")
print("="*60)
selected_samples = create_category_sample_subset(dataset, category_field='raw_subject', min_examples=10, samples_per_category=50)

if selected_samples:
    print(f"\n🎉 Successfully created a sample subset with {len(selected_samples)} examples!")
    
    # Ask if user wants to see details of some samples
    user_input = input("\nDo you want to see detailed view of some selected samples? (y/n): ")
    if user_input.lower() == 'y':
        num_to_show = min(3, len(selected_samples))
        print(f"\nShowing detailed view of first {num_to_show} samples:")
        
        for i in range(num_to_show):
            sample = selected_samples[i]
            print(f"\n{'='*60}")
            print(f"SAMPLE {i+1}: {sample['category']}")
            print(f"{'='*60}")
            display_dataset_item(sample['item_data'], sample['dataset_index'])
            
            if i < num_to_show - 1:
                user_input = input("\nPress Enter to continue to next sample, or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
else:
    print("\n❌ No samples were selected (no categories with >= 5 examples)")

# Ask if user wants to see sample items
user_input = input("\nDo you want to see sample items from the dataset? (y/n): ")
if user_input.lower() == 'y':
    # Display the first 6 items with images
    for i in range(6):
        try:
            item = dataset[i]
            display_dataset_item(item, i)
            
            # Ask user if they want to continue
            user_input = input("\nPress Enter to continue to next item, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

# Check for test case question match
print("\n" + "="*60)
print("CHECKING FOR TEST CASE QUESTION")
print("="*60)
matches = check_for_test_case_question(dataset)

if matches:
    print(f"\n🎉 Found {len(matches)} matching item(s) in the dataset!")
    
    # Ask if user wants to see details of the matches
    user_input = input("\nDo you want to see full details of the matching items? (y/n): ")
    if user_input.lower() == 'y':
        for i, match in enumerate(matches):
            print(f"\n{'='*60}")
            print(f"DETAILED VIEW OF MATCH {i+1}")
            print(f"{'='*60}")
            display_dataset_item(match['item'], match['index'])
            
            if i < len(matches) - 1:
                user_input = input("\nPress Enter to continue to next match, or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
else:
    print("\n❌ No matching questions found in the dataset")

print("\nDone!")