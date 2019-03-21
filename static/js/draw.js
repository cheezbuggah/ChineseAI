window.onload = function() {
        var myCanvas = document.getElementById("myCanvas");
        var Erase = document.getElementById("erase");
        var upload = document.getElementById("upload");
        var curColor = "black";

        if(myCanvas){
                        var isDown = false;
                        var ctx = myCanvas.getContext("2d");
                        var canvasX, canvasY;
                        ctx.fillStyle = "white"
                        ctx.lineWidth = 12;
                        ctx.fillRect(0,0,300,300);
                            
                        $(myCanvas)
                        
                        

                        .mousedown(function(e){
                                        isDown = true;
                                        ctx.beginPath();
                                        canvasX = e.pageX - myCanvas.offsetLeft;
                                        canvasY = e.pageY - myCanvas.offsetTop;
                                        ctx.moveTo(canvasX, canvasY);
                        })
                        .mousemove(function(e){
                                        if(isDown != false) {
                                                canvasX = e.pageX - myCanvas.offsetLeft;
                                                canvasY = e.pageY - myCanvas.offsetTop;
                                                ctx.lineTo(canvasX, canvasY);
                                                ctx.strokeStyle = curColor;
                                                
                                                ctx.stroke();
                                                
                                        }
                        })
                        .mouseup(function(e){
                                        isDown = false;
                                        ctx.closePath();
                        
                        })
                        Erase.onclick = function(){
                            ctx.fillRect(0,0,300,300);
                        };


                }
                        

        upload.onclick = function(){
            var dataURL = myCanvas.toDataURL("image/png");
            $('#successAlert').text(dataURL).show();
            $.ajax({
                data : {
                    imgBase64: dataURL
                },
                type : 'POST',
                url : '/process'
                })
            .done(function(data){
                //ctx.fillRect(0,0,200,300);
            })
        }

};
