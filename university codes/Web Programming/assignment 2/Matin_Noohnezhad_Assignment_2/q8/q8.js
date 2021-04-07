var script = document.createElement('script');
script.src = 'https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js';
document.getElementsByTagName('head')[0].appendChild(script);

function func(elementID) {

    if (elementID == "flower") {
        $("#flower").show();
        $("#b1").css("color", "red");
        $("#car").hide();
        $("#b2").css("color", "black");
        $("#house").hide();
        $("#b3").css("color", "black");
    } else if (elementID == "car") {
        $("#flower").hide();
        $("#b1").css("color", "black");
        $("#car").show();
        $("#b2").css("color", "red");
        $("#house").hide();
        $("#b3").css("color", "black");
    } else if (elementID == "house") {
        $("#flower").hide();
        $("#b1").css("color", "black");
        $("#car").hide();
        $("#b2").css("color", "black");
        $("#house").show();
        $("#b3").css("color", "red");
    }
}

