window.onload = function() {



    //#region Dictionaries
    var chardict = [
        "あ","い","う","え","お",
        "か","き","く","け","こ",
        "さ","し","す","せ","そ",
        "た","ち","つ","て","と",
        "な","に","ぬ","ね","の",
        "は","ひ","ふ","へ","ほ",
        "ま","み","む","め","も",
        "や","ゆ","よ",
        "ら","り","る","れ","ろ",
        "わ","ゐ",　　　"ゑ","を",
        "ん","ゝ"
        ]
    var voiced =
    [
        "","","","","",
        "が","ぎ","ぐ","げ","ご",
        "ざ","じ","ず","ぜ","ぞ",
        "だ","ぢ","づ","で","ど",
        "","","","","",
        "ば","び","ぶ","べ","ぼ",
        "","","","","",
        "","","",
        "","","","","",
        "","","","",
        "","ゞ"
    ]
    var voicednum =
    [
        "","","","","",
        "が","ぎ","ぐ","げ","ご",
        "ざ","じ","ず","ぜ","ぞ",
        "だ","ぢ","づ","で","ど",
        "","","","","",
        "ば","び","ぶ","べ","ぼ",
        "","","","","",
        "","","",
        "","","","","",
        "","","","",
        "","ゞ"
    ]
    var small =
    [
        "ぁ","ぃ","ぅ","ぇ","ぉ",
        "","","","","",
        "","","","","",
        "","","っ","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "ゃ","ゅ","ょ",
        "","","","","",
        "","","","",
        "",""
    ]
    var variation = [
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "ぱ","ぴ","ぷ","ぺ","ぽ",
        "","","","","",
        "","","",
        "","リ","","","",
        "","","","",
        "","〻"
    ]
    var prondict = [
        "a","i","u","e","o",
        "ka","ki","ku","ke","ko",
        "sa","shi","su","se","so",
        "ta","chi","tsu","te","to",
        "na","ni","nu","ne","no",
        "ha","hi","fu","he","ho",
        "ma","mi","mu","me","mo",
        "ya","yu","yo",
        "ra","ri","ru","re","ro",
        "wa","wi","we","wo",
        "n","-"
    ]
    //for voiced
    var prondict_1 = [
        "","","","","",
        "ga","gi","gu","ge","go",
        "za","ji","zu","ze","zo",
        "da","ji","zu","de","do",
        "","","","","",
        "ba","bi","bu","be","bo",
        "","","","","",
        "","","",
        "","","","","",
        "","","","",
        "","-"
    ]
    //for variant and small
    var prondict_2 = [
        "Like their big counterpart: a","Like their big counterpart: i","Like their big counterpart: u","Like their big counterpart: e","Like their big counterpart: o",
        "","","","","",
        "","","ʔ","","",
        "","","","","",
        "","","","","",
        "pa","pi","pu","pe","po",
        "","","","","",
        "Like their big counterpart","Like their big counterpart","Like their big counterpart",
        "","ri","","","",
        "","","","",
        "","-"
    ]
    var notes = [
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","Note: This can be pronounced 'e' when used as a particle","",
        "","","","","",
        "","","",
        "","","","","",
        "Note: It can be pronounced 'ha' when used as a particle","Note: Obsolete","Note: Obsolete","Note: It can be pronounced 'o' when used as a particle",
        "Note: The only formal final vowel in Japanese","Note: This character repeats and devoices the previous sound, but it fell out of use."
    ]
    //for voiced
    var notes_1 = [
        "","","","","",
        "","","","","",
        "","Note: Keep in mind that there are two characters with the sound 'ji', the other one being ぢ, but じ is more common.","Note: Keep in mind that there are two characters with the sound 'zu', the other one being ず, which is more common.","","",
        "","Note: Keep in mind that there are two characters with the sound 'ji', the other one being じ, which is the more common one","Note: Keep in mind that there are two characters with the sound 'zu', the other one being づ, but じ is more common.","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","",
        "","Note: This character repeats and voices the previous sound, but it fell out of use."
    ]
    //for variant and small
    var notes_2 = [
        "Note: Only used in digraphs.","Note: Only used in digraphs.","Note: Only used in digraphs","Note: Only used in digraphs.","Note: Only used in digraphs.",
        "","","","","",
        "","","","This is used in front of a character to double the next consonant:<br>ここ - koko ➞ こっこ - kokko<br>This is not used when the consonant is a n-; then the ん is used.<br>まに - mani ➞ まんに - manni","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "","","","","",
        "Note: This character is used after character with the 'i' vowel except for い.<br>きゃ ➞ kya","Note: This character is used after character with the 'i' vowel except for い.<br>きゅ　➞ kyu","Note: This character is used after character with the 'i' vowel except for い.<br>きょ　➞ kyo",
        "","Note: Make sure you're not confusing it with い, where the right stroke is shorter than the left.<br> If this is still the right character, then it's a variation of り since the two strokes aren't always bridged.<br>If you're copying and pasting, make sure to use り instead. ","","","",
        "","","","",
        "","Note: This character repeats the previous sound in vertical writing, but it fell out of use."
    ]
    //#endregion Dictionaries

    var myCanvas = document.getElementById("myCanvas");
    var curColor = "black";
    var results = [];
    var border = parseInt($("#myCanvas").css("border-width"))*2;
    var debug = false;
    var candidate;
    var val = 10;


    if (debug)
        $(".debug").show();
    else
        $(".debug").hide();

    if(myCanvas){
                    var isDown = false;
                    var ctx = myCanvas.getContext("2d");
                    var canvasX, canvasY;
                    ctx.fillStyle = "white"
                    ctx.lineWidth = val;
                    ctx.fillRect(0,0,300,300);

                    $(myCanvas)
                    .mousedown(function(e){
                                    ctx.lineWidth = val;
                                    isDown = true;
                                    ctx.beginPath();
                                    canvasX = e.pageX - myCanvas.offsetLeft-border*((e.pageX - myCanvas.offsetLeft)/myCanvas.offsetWidth);
                                    canvasY = e.pageY - myCanvas.offsetTop-border*((e.pageX - myCanvas.offsetLeft)/myCanvas.offsetHeight);
                                    ctx.moveTo(canvasX, canvasY);
                    })
                    .mousemove(function(e){
                                    if(isDown != false) {
                                            ctx.lineWidth = val;
                                            canvasX = e.pageX - myCanvas.offsetLeft-border*((e.pageX - myCanvas.offsetLeft)/myCanvas.offsetWidth);
                                            canvasY = e.pageY - myCanvas.offsetTop-border*((e.pageY - myCanvas.offsetTop)/myCanvas.offsetHeight);
                                            ctx.lineTo(canvasX, canvasY);
                                            ctx.strokeStyle = curColor;

                                            ctx.stroke();

                                    }
                    })
                    .mouseup(function(e){
                                    isDown = false;
                                    ctx.closePath();

                    })
                    $("#erase").click(function(){
                        ctx.fillRect(0,0,300,300);
                    });

            }


        $("#upload").click(function(){
            var dataURL = myCanvas.toDataURL("image/png");
            console.log(dataURL);
            //$('#successAlert').text(dataURL).show();
            $.ajax({
                data : {
                    imgBase64: dataURL
                },
                type : 'POST',
                url : '/process'
                })
            .done(function(data){
                //forresults();
                results = data.rarray;
                //results = reresult(results);
                getmax();
            })
            event.preventDefault();
        });

    resultshow = function(){
        $('.results').show();
        $("#value").html(chardict[candidate]);
        $("#pron").html('It is pronounced: '+prondict[candidate]);
        $("#note").html(notes[candidate]);
        $("#yesb").show();
        $("#nob").show();

        //variation buttons
        if(voiced[candidate]!="")
            $("#var_1").html("It actually was "+voiced[candidate]).show();
        if(variation[candidate]!="")
            $("#var_2").html("It actually was "+variation[candidate]).show();
        if(small[candidate]!="")
            $("#var_3").html("Yes, but smaller").show();



        if(debug){
            debugdiv = document.getElementById("debugresults");
            results.forEach(function(value, index){
                debugdiv.innerHTML = debugdiv.innerHTML + index+ ": "+value+" "+chardict[index]+" c: "+candidate+"<br>"
            })
        }
    }

    //#region button.click()
    $("#var_1").click(function(){
        save(voiced[candidate]);
        $("#value").html(voiced[candidate]);
        $("#pron").html('It is pronounced: '+prondict_1[candidate]);
        $("#note").html(notes_1[candidate]+" ");
        $("#var_1").hide();
        $("#var_2").hide();
        $("#var_3").hide();
        $("#yesb").hide();
        $("#nob").hide();
    });

    $("#var_2").click(function(){
        save(variation[candidate]);
        $("#value").html(variation[candidate]);
        $("#pron").html('It is pronounced: '+prondict_2[candidate]);
        $("#note").html(notes_2[candidate]);
        $("#var_1").hide();
        $("#var_2").hide();
        $("#var_3").hide();
        $("#yesb").hide();
        $("#nob").hide();
    });
    $("#var_3").click(function(){
        save(small[candidate]);
        $("#value").html(small[candidate]);
        $("#pron").html('It is pronounced: '+prondict_2[candidate]);
        $("#note").html(notes_2[candidate]);
        $("#var_1").hide();
        $("#var_2").hide();
        $("#var_3").hide();
        $("#yesb").hide();
        $("#nob").hide();
    });
    $("#nob").click(function(){
        results[candidate]=0;
        $("#var_1").hide();
        $("#var_2").hide();
        $("#var_3").hide();
        $("#debugresults").html("");
        getmax();
    });
    $("#yesb").click(function(){
        save(chardict[candidate]);
        $("#var_1").hide();
        $("#var_2").hide();
        $("#var_3").hide();
        $("#yesb").hide();
        $("#nob").hide();
    })
    save = function(c){
        $.ajax({
            data : {
                jchar: c
            },
            type : 'POST',
            url : '/chosen'
            })
        event.preventDefault();
    }

    getmax = function(){
        var max = 0;
        var place = 48;
        results.forEach(function(value, index){
            if(max<value){
                max = value;
                place = index;
            }
        })
        candidate = place;
        resultshow();
    }
    var slider = document.getElementById("brush");
    slider.oninput = function(){
        val = slider.value/5
        $("#brushv").html(val)
    }
    reresult = function(a){
        var n = [];
        for(var i = 0; i<a.length;i++){
            n[i] = Number(a[i]);
            console.log(a[i]);
        }
        return n;
    }

    //#endregion button.click()


    //Creates random results for debug purposes only
    forresults = function(){
        results = [];
        for(var i = 0; i<49;i++){
            results.push(Math.random());
        }
    }


};
