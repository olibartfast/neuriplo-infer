

import argparse
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser('RF-DETR Detection LibTorch Export Script',
                                     description='Export RF-DETR detection model to TorchScript format for LibTorch C++')

    # Export options
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save exported model (default: current directory)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='Input image size (default: 640)')
    parser.add_argument('--model_type', default='base', type=str,
                        choices=['base', 'nano', 'small', 'medium', 'large'],
                        help='Model type (default: base)')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to model checkpoint to load weights from')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize model for inference')
    parser.add_argument('--num_classes', default=91, type=int,
                        help='Number of classes (default: 91 for COCO)')
    parser.add_argument('--num_queries', default=300, type=int,
                        help='Number of detection queries (default: 300)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RF-DETR Detection Model LibTorch Export")
    print("="*60)

    # Initialize the detection model
    print("\n[1/3] Loading RF-DETR Detection model...")
    model_wrapper = None
    if args.model_type == 'base':
        from rfdetr import RFDETRBase
        model_wrapper = RFDETRBase()
    elif args.model_type == 'nano':
        from rfdetr import RFDETRNano
        model_wrapper = RFDETRNano()
    elif args.model_type == 'small':
        from rfdetr import RFDETRSmall
        model_wrapper = RFDETRSmall()
    elif args.model_type == 'medium':
        from rfdetr import RFDETRMedium
        model_wrapper = RFDETRMedium()
    elif args.model_type == 'large':
        from rfdetr import RFDETRLarge
        model_wrapper = RFDETRLarge()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    print(f"  - Wrapper type: {type(model_wrapper).__name__}")
    
    # Get the actual torch.nn.Module
    print("  - Extracting torch.nn.Module...")
    if not hasattr(model_wrapper, 'model'):
        raise ValueError("Wrapper doesn't have 'model' attribute")
    
    model_instance = model_wrapper.model
    
    # Find the actual nn.Module
    model = None
    if hasattr(model_instance, 'model') and isinstance(model_instance.model, torch.nn.Module):
        model = model_instance.model
        print(f"    ✓ Found torch.nn.Module: {type(model).__name__}")
    else:
        raise ValueError("Could not find torch.nn.Module")
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\n  - Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("    ✓ Checkpoint loaded")
    
    # Set model to eval mode
    model.eval()
    print("  ✓ Model set to eval mode")
    
    # Create a tracing wrapper as a simple callable
    print("\n  - Creating tracing wrapper...")
    
    class TracingWrapper(nn.Module):
        """Simple wrapper that only traces the forward pass"""
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, x):
            # Call the model
            with torch.no_grad():
                outputs = self.base_model(x)
            
            # Extract outputs
            logits = outputs['pred_logits'].contiguous()
            boxes = outputs['pred_boxes'].contiguous()
            
            # Return as tuple
            return logits, boxes
    
    wrapper = TracingWrapper(model)
    wrapper.eval()
    
    print(f"    ✓ Wrapper created (num_queries={args.num_queries}, num_classes={args.num_classes})")
    
    # Export to TorchScript
    print("\n[2/3] Exporting to TorchScript format...")
    print(f"  - Model type: {args.model_type}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Input size: {args.input_size}x{args.input_size}")
    
    # Create dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size)
    
    # Test forward pass
    print("\n  - Testing forward pass...")
    try:
        with torch.no_grad():
            test_output = wrapper(dummy_input)
        if isinstance(test_output, tuple):
            print(f"    ✓ Forward pass successful (tuple with {len(test_output)} elements)")
            for i, t in enumerate(test_output):
                print(f"      Output {i}: {t.shape}")
        else:
            print(f"    ✓ Forward pass successful")
            print(f"      Output type: {type(test_output)}")
    except Exception as e:
        print(f"    ✗ Forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Trace only the forward method - this avoids recursive scripting
    print(f"\n  - Tracing forward method only (avoids recursive scripting)...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, dummy_input, strict=False)
        print("    ✓ Trace successful")
    except Exception as e:
        print(f"    ✗ Trace failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test traced model
    print(f"\n  - Testing traced model...")
    try:
        with torch.no_grad():
            output = traced_model(dummy_input)
        if isinstance(output, tuple):
            print(f"    ✓ Test successful (tuple with {len(output)} elements)")
            for i, t in enumerate(output):
                print(f"      Output {i}: {t.shape}")
        else:
            print(f"    ! Output is {type(output)}, not tuple")
    except Exception as e:
        print(f"    ✗ Test failed: {str(e)}")
    
    # Skip freeze
    print("\n  - Skipping freeze to preserve output structure")
    
    # Optimize if requested  
    if args.optimize:
        print("\n  - Optimizing for inference...")
        try:
            traced_model = torch.jit.optimize_for_inference(traced_model)
            print("    ✓ Optimization successful")
        except Exception as e:
            print(f"    ! Optimization failed: {str(e)}")
    
    # Save
    print("\n[3/3] Saving TorchScript model...")
    output_dir = args.output_dir if args.output_dir else '.'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'rfdetr_{args.model_type}_torchscript.pt')
    traced_model.save(output_path)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ File size: {file_size:.2f} MB")
    
    # Verify
    print("\n[Verification] Loading and testing saved model...")
    try:
        loaded_model = torch.jit.load(output_path)
        loaded_model.eval()
        
        with torch.no_grad():
            output = loaded_model(dummy_input)
        
        if isinstance(output, tuple):
            print(f"  ✓ Model loaded successfully (tuple with {len(output)} elements)")
            for i, t in enumerate(output):
                print(f"    Output {i}: shape={t.shape}, dtype={t.dtype}")
        else:
            print(f"  ! Output is {type(output)}")
            
    except Exception as e:
        print(f"  ✗ Verification failed: {str(e)}")
    
    # Debug graph structure
    print("\n[Debug] Analyzing TorchScript graph...")
    try:
        loaded = torch.jit.load(output_path)
        graph = loaded.graph
        outputs = list(graph.outputs())
        
        print(f"  Number of graph outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            out_type = out.type()
            print(f"\n  Output {i} (debugName={out.debugName()}):")
            print(f"    Type: {out_type}")
            print(f"    Type str: {str(out_type)}")
            
            # Check what it is
            try:
                if hasattr(out_type, 'kind'):
                    print(f"    Type kind: {out_type.kind()}")
            except:
                pass
            
            # Try casting to different types
            try:
                tensor_type = out_type.cast(torch._C.TensorType)
                if tensor_type:
                    print(f"    ✓ Is TensorType")
                    try:
                        sizes = tensor_type.sizes()
                        if sizes and hasattr(sizes, 'concrete_sizes') and sizes.concrete_sizes():
                            print(f"      Shape: {sizes.concrete_sizes()}")
                    except:
                        pass
                else:
                    print(f"    ✗ Not a TensorType")
            except Exception as e:
                print(f"    ! TensorType check failed: {e}")
            
            try:
                tuple_type = out_type.cast(torch._C.TupleType)
                if tuple_type:
                    print(f"    ✓ Is TupleType")
                    elements = tuple_type.elements()
                    print(f"      Elements: {len(elements)}")
                    for j, elem in enumerate(elements):
                        print(f"        Element {j}: {elem}")
                        try:
                            elem_tensor = elem.cast(torch._C.TensorType)
                            if elem_tensor:
                                elem_sizes = elem_tensor.sizes()
                                if elem_sizes and hasattr(elem_sizes, 'concrete_sizes') and elem_sizes.concrete_sizes():
                                    print(f"          Shape: {elem_sizes.concrete_sizes()}")
                        except:
                            pass
                else:
                    print(f"    ✗ Not a TupleType")
            except Exception as e:
                print(f"    ! TupleType check failed: {e}")
                
    except Exception as e:
        print(f"  ! Graph analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✓ LibTorch Export Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the [Debug] output above")
    print("2. Update C++ code to handle the detected output type")
    print("="*60)


if __name__ == '__main__':
    main()