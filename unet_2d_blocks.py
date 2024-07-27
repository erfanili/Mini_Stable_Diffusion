
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #251
        def get_mid_block(
            mid_block_type: str,
            in_channels: int,
            
        ):
            if mid_block_type == "UNetMidBlock2DCrossAttn":
                return UNetMidBlock2DCrossAttn(
                    
                )
        #307
            elif mid_block_type =="UNetMidBlock2D":
                return UNetMidBlock2D(
                    in_channels = in_channels,
            )
            
        
        
        
        
        #326
        def get_up_block(
            up_block_type: str,
            num_layers: int,
        ):
            
        #362
        
            if up_block_type =="UpBlock2D":
                return UpBlock2D(
                    num_layers = num_layers,
                )
                
            elif up_block_type == "CrossAttnUpBlock2D":
                
                return CrossAttnUpBlock2D(
                    num_layers = num_layers,
                )
                    
                #377
            elif up_block_type =="ResnetUpsampleBlock2D":
                return ResnetUpsampleBlock2D(
                    num_layers = num_layers,
                )
                        
            elif up_block_type =="CrossAttnUpBlock2D":
                
                return CrossAttnUpBlock2D
            
            
            elif up_block_type = "AttnUpBlock2D":
                
                return AttnUpBlock2D(
                    num_layers = num_layers,
                    in_channels = in_channels,
                    out_channels = out_channels,
                )
            
        
        