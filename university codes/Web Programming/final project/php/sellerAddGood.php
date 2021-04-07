<?php
session_start();
//
//
$sellerId = $_SESSION['sellerId'];
//
$goodNameError = "";
//
$servername = "localhost";
$user = "root";
$password = "";
$dbname = "final_project";

// Create connection
$conn = new mysqli($servername, $user, $password, $dbname);
mysqli_set_charset($conn, "utf8");
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$exist = false;
$sql = "SELECT categoryName FROM categories";
$result = $conn->query($sql);
$categories = "";

if ($result->num_rows > 0) {
    // output data of each row
    while ($row = $result->fetch_assoc()) {
        $categories .= "<option value='" . $row["categoryName"] . "'>";
        $categories .= $row["categoryName"];
        $categories .= "</option>";
//        <option value="volvo">category1</option>
    }
}
//check validation in database
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $id = $_POST["id"];
    $name = test_input($_POST["name"]);
    $price = $_POST["price"];
//    $picture = test_input($_POST["picture"]);
    $category = test_input($_POST["category"]);
    $description = test_input($_POST["description"]);

//    $servername = "localhost";
//    $user = "root";
//    $password = "";
//    $dbname = "final_project";

// Create connection
//    $conn = new mysqli($servername, $user, $password, $dbname);
//    mysqli_set_charset($conn, "utf8");
// Check connection
//    if ($conn->connect_error) {
//        die("Connection failed: " . $conn->connect_error);
//    }

    $exist = false;
    $sql = "SELECT name,store_id FROM good";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // output data of each row
        while ($row = $result->fetch_assoc()) {
            if ($row["name"] == $name) {
                if ($row["store_id"] == $id) {
                    $exist = true;
                    break;
                }
            }
        }
    }
    if ($exist == true) {
        $goodNameError = 'شما از قبل یک محصول با این اسم ثبت کرده اید.';
    } else {
//        $category_id = 0;
//        $sql = "SELECT id FROM categories where categoryName='" . $category . "'";
//        $result = $conn->query($sql);
//        if ($result->num_rows > 0) {
//             output data of each row
//            while ($row = $result->fetch_assoc()) {
//                $category_id = $row["id"];
//            }
//        }
        //
        $image_dir = "../good_images/";
        $image_dir .= $id;
        $image_dir .= 'xxx';
        $image_dir .= $name;
        $image_dir = $image_dir . basename($_FILES["picture"]["name"]);
        $uploadOk = 1;
        $imageFileType = strtolower(pathinfo($image_dir, PATHINFO_EXTENSION));
        // Check if image file is a actual image or fake image
        if (isset($_POST["submit"])) {
            $check = getimagesize($_FILES["picture"]["tmp_name"]);
            if ($check !== false) {
//                echo "File is an image - " . $check["mime"] . ".";
                $uploadOk = 1;
            } else {
                echo "File is not an image.";
                $uploadOk = 0;
            }
        }
        // Check if file already exists
        if (file_exists($image_dir)) {
            echo "Sorry, file already exists.";
            $uploadOk = 0;
        }
        // Check file size
        if ($_FILES["picture"]["size"] > 20000000) {
            echo "Sorry, your file is too large.";
            $uploadOk = 0;
        }
        // Allow certain file formats
        if ($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg"
            && $imageFileType != "gif") {
            echo "Sorry, only JPG, JPEG, PNG & GIF files are allowed.";
            $uploadOk = 0;
        }
        // Check if $uploadOk is set to 0 by an error
        if ($uploadOk == 0) {
            echo "Sorry, your file was not uploaded.";
            // if everything is ok, try to upload file
        } else {
            if (move_uploaded_file($_FILES["picture"]["tmp_name"], $image_dir)) {
//                echo "The file " . basename($_FILES["picture"]["name"]) . " has been uploaded.";
            } else {
                echo "Sorry, there was an error uploading your file.";
            }
        }

        $sql = "INSERT INTO good (name, categoryName,image_dir,price,store_id,description,confirmed)
VALUES ('" . $name . "','" . $category . "','" . $image_dir . "'," . $price . "," . $id . ",'" . $description . " ',0)";
        if ($conn->query($sql) === TRUE) {
            echo 'insert to good successfully';
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
    }
    $conn->close();
}


//

$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>تقاضای ثبت محصول</title>
    <link rel="stylesheet" href="../css/sellerAddGood.css">
</head>
<body>
<form method="post" action="sellerAddGood.php" enctype="multipart/form-data" name="myForm" onsubmit="return formValidation()">
    <fieldset>
        <legend>تقاضای ثبت محصول</legend>
        <input class="input" type="number" name="id" style="display: none" value="' . $sellerId . '">
        <br>
        نام محصول جدید:
        <br>
        <input class="input" type="text" name="name">
        <p class="error" id="err1">' . $goodNameError . '</p>
        <br>
        <br>
        قیمت محصول (به تومان):
        <br>
        <input class="input" type="number" min="0.00" max="1000000000.00" step="1" name="price">
        <p class="error" id="err2"></p>
        <br>
        <br>
        عکس محصول:
        <br>
        <input onchange="fileSelect()" class="input" type="file" name="picture">
        <p class="error" id="err3"></p>
        <br>
        <br>
        دسته بندی محصول:
        <br>
        <!--        <input class="input" type="text" name="category">-->
        <select class="input" name="category">
            ' . $categories . '
        </select>
        <br>
        <br>
        توضیحات(اختیاری):
        <br>
        <textarea class="input" name="description" rows="8" cols="50"></textarea>
        <br>
        <br>
        <input id="submit" type="submit" name="submit" value="تقاضای ثبت">
        <br>
    </fieldset>
</form>

<script src="../js/sellerAddGood.js"></script>
</body>
</html>';

echo $string;

function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

?>
