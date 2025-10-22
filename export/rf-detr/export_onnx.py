import argparse


def main():
    parser = argparse.ArgumentParser('RF-DETR Detection Export Script',
                                     description='Export RF-DETR detection model to ONNX format')

    # Export options that will be passed to the model's export() method
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save exported model (default: current directory)')
    parser.add_argument('--opset_version', default=17, type=int,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model using onnxsim')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='Input image size (default: 640)')
    parser.add_argument('--model_type', default='base', type=str,
                        choices=['base', 'nano', 'small', 'medium', 'large'],
                        help='Model type (default: base)')
    args = parser.parse_args()
    
    print("="*60)
    print("RF-DETR Detection Model Export")
    print("="*60)

    # Initialize the detection model
    print("\n[1/2] Loading RF-DETR Detection model...")
    model = None
    if args.model_type == 'base':
        from rfdetr import RFDETRBase
        model = RFDETRBase()
    elif args.model_type == 'nano':
        from rfdetr import RFDETRNano
        model = RFDETRNano()
    elif args.model_type == 'small':
        from rfdetr import RFDETRSmall
        model = RFDETRSmall()
    elif args.model_type == 'medium':
        from rfdetr import RFDETRMedium
        model = RFDETRMedium()
    elif args.model_type == 'large':
        from rfdetr import RFDETRLarge
        model = RFDETRLarge()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
        
    # Build export kwargs from arguments
    export_kwargs = {
        'opset_version': args.opset_version,
        'simplify': args.simplify,
        'batch_size': args.batch_size,
    }
    
    # Add output_dir if specified
    if args.output_dir:
        export_kwargs['output_dir'] = args.output_dir
    
    # Export using the model's built-in export method
    print("\n[2/2] Exporting to ONNX format...")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Input size: {args.input_size}x{args.input_size}")
    print(f"  - ONNX opset: {args.opset_version}")
    print(f"  - Simplify: {args.simplify}")
    
    model.export(**export_kwargs)
    
    print("\n" + "="*60)
    print("✓ Export complete!")
    print("="*60)
    print("\nModel outputs:")
    print("  - dets: Bounding boxes [batch, num_queries, 4]")
    print("  - labels: Class logits [batch, num_queries, num_classes]")
    print("\nNote: This is a detection model.")
    print("="*60)


if __name__ == '__main__':
    main()       