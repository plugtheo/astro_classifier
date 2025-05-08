# Process all images in the test input directory
$inputDir = "data/test_input"
$outputDir = "data/test_output"
$modelPath = "checkpoints/astro-classifier-epoch=09-val_loss=0.32.ckpt"  # Best model based on validation loss

# Get all jpg files
$images = Get-ChildItem -Path $inputDir -Filter "*.jpg"

foreach ($image in $images) {
    $inputPath = Join-Path $inputDir $image.Name
    $outputPath = Join-Path $outputDir "vis_$($image.Name)"
    
    Write-Host "`nProcessing $($image.Name)..."
    Write-Host "Using model: $modelPath"
    python classify.py $inputPath --model_path $modelPath --save_visualization $outputPath --confidence_threshold 0.2
    Write-Host "Saved visualization to $outputPath`n"
} 