function func(elementID) {

    if (elementID == "flower") {
        document.getElementById("flower").style.display = "block";
        document.getElementById("b1").style.color = "red";
        document.getElementById("car").style.display = "none";
        document.getElementById("b2").style.color = "black";
        document.getElementById("house").style.display = "none";
        document.getElementById("b3").style.color = "black";
    } else if (elementID == "car") {
        document.getElementById("flower").style.display = "none";
        document.getElementById("b1").style.color = "black";
        document.getElementById("car").style.display = "block";
        document.getElementById("b2").style.color = "red";
        document.getElementById("house").style.display = "none";
        document.getElementById("b3").style.color = "black";
    } else if (elementID == "house") {
        document.getElementById("flower").style.display = "none";
        document.getElementById("b1").style.color = "black";
        document.getElementById("car").style.display = "none";
        document.getElementById("b2").style.color = "black";
        document.getElementById("house").style.display = "block";
        document.getElementById("b3").style.color = "red";
    }
}

