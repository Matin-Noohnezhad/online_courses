function myFunc2(color, color2, color3) {
    elements = document.getElementsByClassName("d3");
    for (var i = 0; i < elements.length; i++) {
        elements[i].style.backgroundColor = color;
    }
    elements2 = document.getElementsByClassName("d4");
    for (var i = 0; i < elements2.length; i++) {
        elements2[i].style.backgroundColor = color2;
    }
    elements3 = document.getElementsByClassName("d5");
    for (var i = 0; i < elements3.length; i++) {
        elements3[i].style.backgroundColor = color3;
    }
}

function getRandomColor() {
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

function myFunc() {
    window.setInterval(function () {
        // window.alert("matin")
        var color = getRandomColor();
        var color2 = getRandomColor();
        var color3 = getRandomColor();
        myFunc2(color, color2, color3);
    }, 3000);

}
